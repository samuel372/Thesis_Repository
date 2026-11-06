"""Retrieve CRSP financial records and Sustainalytics overall ESG Risk Scores.

Credentials are supplied by WRDS_USERNAME and the user's normal .pgpass file.
Only overall Sustainalytics ESG Risk Score is collected. Lower scores are better.
"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from .data import load_panel, checked_range
import wrds

# Vendor identifiers are supplied through ExperimentConfig. Collection verifies
# the selected field against the WRDS catalogue and records query provenance.


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def local_path(cfg, value):
    path = Path(value)
    path = (path if path.is_absolute() else cfg.root/path).resolve()
    if not path.is_relative_to(cfg.root.resolve()):
        raise ValueError("Data manifests must be inside MAIN_NOTEBOOK_FOLDER.")
    return path


def align_scores(financial, selection, archives, cfg):
    """Use the latest archive already available at the decision date, never a future score."""
    panel = financial.merge(selection[["permno", "entityid"]], on="permno", validate="many_to_one")
    panel["date"] = pd.to_datetime(panel.date)
    h = archives.copy()
    h["fieldvalue"] = pd.to_numeric(h.fieldvalue, errors="coerce")
    h["file_date"], h["fielddate"] = pd.to_datetime(h.file_date), pd.to_datetime(h.fielddate)
    h = h.dropna(subset=["fieldvalue", "fielddate", "file_date"])
    h = h[h.fieldvalue.ge(0) & h.fielddate.le(h.file_date)]
    h["available_date"] = h.file_date+BDay(cfg.publication_lag_business_days)
    h = h.sort_values(["entityid", "available_date", "fielddate"])
    latest = h.groupby(["entityid", "available_date"]).fielddate.transform("max")
    h = h[h.fielddate.eq(latest)]
    if h.groupby(["entityid", "available_date"]).fieldvalue.nunique().gt(1).any():
        raise ValueError("Conflicting ESG values for one entity/archive date.")
    h = h.drop_duplicates(["entityid", "available_date"], keep="last")
    aligned = []
    for entity, group in panel.groupby("entityid", sort=False):
        history = h[h.entityid.eq(entity)]
        out = pd.merge_asof(group.sort_values("date"),
            history[["available_date", "file_date", "fieldvalue"]].sort_values("available_date"),
            left_on="date", right_on="available_date", direction="backward")
        age = (out.date-out.file_date).dt.days
        out["overall_esg_risk"] = out.fieldvalue.where(age.between(0, cfg.maximum_score_age_days))
        out["esg_archive_date"], out["esg_available_date"], out["esg_age_days"] = out.file_date, out.available_date, age
        aligned.append(out.drop(columns=["available_date", "file_date", "fieldvalue"]))
    return pd.concat(aligned, ignore_index=True).rename(columns={"entityid": "sustainalytics_entityid"})


def download_basket(db, cfg, seed, collection):
    """Randomise the eligible universe with the stock seed, then align real scores."""
    keys = ["financial_start", "esg_start", "end_date", "number_of_assets", "candidate_pool_size",
            "minimum_past_observations", "maximum_score_age_days", "publication_lag_business_days"]
    data_settings = {k: getattr(cfg, k) for k in keys}
    data_settings.update(selection_seed=int(seed), financial_table=cfg.financial_table,
                         score_table=f"{cfg.sustainability_library}.{cfg.esg_table}", score_field=cfg.esg_field_id)
    fingerprint = sha256(json.dumps(data_settings, sort_keys=True).encode()).hexdigest()[:12]
    cache = cfg.root/"data"/"cache"
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache/f"basket_{fingerprint}.json"
    if cached.exists() and not cfg.refresh_wrds:
        previous = json.loads(cached.read_text())
        path = cfg.root/previous["panel"]
        if previous["settings"] != data_settings or sha256(path.read_bytes()).hexdigest() != previous["sha256"]:
            raise ValueError("Local cached panel changed. Refresh WRDS rather than silently reusing it.")
        print(f"Local cache: stock seed {seed}")
        return path, "local_cache"

    formation = pd.Timestamp(cfg.esg_start)-BDay(cfg.publication_lag_business_days)
    begin, end = pd.Timestamp(cfg.financial_start), pd.Timestamp(cfg.end_date)
    queries = {}
    queries["eligible_universe"] = f"""
      WITH eligible AS (
        SELECT permno,COUNT(DISTINCT dlycaldt) AS observations FROM {cfg.financial_table}
        WHERE dlycaldt BETWEEN %(start)s AND %(formation)s
          AND dlyret IS NOT NULL AND dlyret > -1 AND dlyprc IS NOT NULL
          AND sharetype='NS' AND securitytype='EQTY' AND securitysubtype='COM'
          AND usincflg='Y' AND issuertype IN ('ACOR','CORP')
          AND primaryexch IN ('N','A','Q') AND conditionaltype='RW' AND tradingstatusflg='A'
        GROUP BY permno HAVING COUNT(DISTINCT dlycaldt) >= %(minimum)s)
      SELECT e.permno::integer,d.cusip,d.ticker FROM eligible e JOIN LATERAL (
        SELECT cusip,ticker FROM {cfg.financial_table} d WHERE d.permno=e.permno
          AND d.dlycaldt BETWEEN %(identifier_start)s AND %(formation)s AND d.cusip IS NOT NULL
        ORDER BY d.dlycaldt DESC LIMIT 1) d ON TRUE ORDER BY e.permno"""
    candidates = db.raw_sql(queries["eligible_universe"], params=dict(start=begin.date(),
        formation=formation.date(), minimum=cfg.minimum_past_observations,
        identifier_start=(formation-pd.Timedelta(days=90)).date()))
    if candidates.empty:
        raise ValueError("No stocks satisfy formation-date eligibility.")
    pool = candidates.iloc[np.random.default_rng(seed).permutation(len(candidates))].head(cfg.candidate_pool_size).copy()
    pool["cusip8"] = pool.cusip.astype(str).str.upper().str[:8]
    queries["rating_links"] = f"""
      SELECT upper(substr(cusip,1,8)) AS cusip8,entityid::bigint,extractiondate
      FROM {cfg.sustainability_library}.reference_data WHERE extractiondate <= %(formation)s
      AND upper(substr(cusip,1,8)) IN %(cusips)s"""
    links = db.raw_sql(queries["rating_links"], params=dict(formation=formation.date(),
        cusips=tuple(pool.cusip8.unique())), date_cols=["extractiondate"])
    if links.empty:
        raise ValueError("No financial/rating identifier links known at formation.")
    links = links[links.extractiondate.eq(links.groupby("cusip8").extractiondate.transform("max"))]
    ambiguity = links.groupby("cusip8").entityid.nunique()
    links = links[~links.cusip8.isin(ambiguity[ambiguity.gt(1)].index)].drop_duplicates("cusip8")
    selected = pool.merge(links[["cusip8", "entityid"]], on="cusip8", how="inner", sort=False)
    entities = tuple(int(x) for x in selected.entityid.unique())
    if not entities:
        raise ValueError("No unambiguous financial/rating identifier links.")
    queries["esg_archives"] = f"""
      SELECT entityid::bigint AS entityid,fieldvalue,fielddate,file_date
      FROM {cfg.sustainability_library}.{cfg.esg_table} WHERE fieldid=%(field)s
        AND entityid IN %(entities)s AND file_date BETWEEN %(lookback)s AND %(end)s
      ORDER BY entityid,file_date,fielddate"""
    archives = db.raw_sql(queries["esg_archives"], params=dict(field=cfg.esg_field_id, entities=entities,
        lookback=(begin-pd.Timedelta(days=cfg.maximum_score_age_days)).date(), end=end.date()),
        date_cols=["fielddate", "file_date"])
    if archives.empty:
        raise ValueError("No ESG archives for these candidates.")
    numeric = pd.to_numeric(archives.fieldvalue, errors="coerce")
    known = archives[archives.file_date.le(formation)
        & archives.file_date.ge(formation-pd.Timedelta(days=cfg.maximum_score_age_days))
        & numeric.notna() & numeric.ge(0)]
    selected = selected[selected.entityid.isin(known.entityid)].drop_duplicates("entityid").head(cfg.number_of_assets)
    if len(selected) != cfg.number_of_assets:
        raise ValueError("Too few stocks with required scores known at formation.")
    queries["financial_returns"] = f"""
      SELECT permno::integer,ticker,dlycaldt AS date,ABS(dlyprc)::double precision AS price,
        dlyret::double precision AS return FROM {cfg.financial_table}
      WHERE permno IN %(permnos)s AND dlycaldt BETWEEN %(start)s AND %(end)s
        AND conditionaltype='RW' AND tradingstatusflg='A' ORDER BY dlycaldt,permno"""
    financial = db.raw_sql(queries["financial_returns"], params=dict(permnos=tuple(int(x) for x in selected.permno),
        start=begin.date(), end=end.date()), date_cols=["date"])
    panel = align_scores(financial, selected, archives, cfg)
    stem = f"seed{seed}_{fingerprint}"
    raw = collection/"raw"
    raw.mkdir(exist_ok=True)
    for name, frame in [("financial", financial), ("selection", selected), ("esg_archives", archives)]:
        frame.to_csv(raw/f"{stem}_{name}.csv", index=False)
    path = collection/f"{stem}.csv"
    panel.to_csv(path, index=False)
    metadata = dict(settings=data_settings, panel=str(path.relative_to(cfg.root)),
        sha256=sha256(path.read_bytes()).hexdigest(), fetched_utc=timestamp(), source="WRDS",
        queries=queries, note="Past-only formation selection, followed by full-period coverage checks.")
    path.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2))
    cached.write_text(json.dumps(metadata, indent=2))
    return path, "wrds_query"


def collect_data(cfg):
    """Return panel paths from WRDS collection or a saved local data manifest."""
    cfg.validate()
    if cfg.data_source == "saved":
        if cfg.saved_manifest is None:
            raise ValueError("Supply this folder's data_manifest.json for deliberate saved-data reuse.")
        path = local_path(cfg, cfg.saved_manifest)
        manifest = json.loads(path.read_text())
        return {role: [cfg.root/p for p in paths] for role, paths in manifest["files"].items()}
    from dotenv import load_dotenv
    load_dotenv(cfg.root/".env", override=False)
    username = os.getenv("WRDS_USERNAME")
    if not username:
        raise ValueError("Set WRDS_USERNAME in this folder's .env. See README.md.")
    
    folder = cfg.root/"data"/"collections"/timestamp()
    folder.mkdir(parents=True, exist_ok=False)
    files = {role: [] for role in cfg.baskets}
    used, entities, attempts, records = set(), set(), 0, []
    seed = cfg.first_stock_seed
    print("WRDS collection:", folder)
    print("Refresh requested:", cfg.refresh_wrds, "| stock seed:", seed)
    with wrds.Connection(wrds_username=username) as db:
        metadata = db.raw_sql(f"SELECT fieldid::bigint,fieldname,description FROM "
            f"{cfg.sustainability_library}.esgrr_data_catalogue WHERE fieldid=%(field)s",
            params=dict(field=cfg.esg_field_id))
        if len(metadata) != 1 or str(metadata.iloc[0].fieldname).strip() != "ESG Risk Score":
            raise ValueError("WRDS catalogue does not match the configured overall ESG Risk Score.")
        metadata.to_csv(folder/"verified_esg_field.csv", index=False)
        for role, needed in cfg.baskets.items():
            while len(files[role]) < needed and attempts < cfg.maximum_collection_attempts:
                stock_seed, seed, attempts = seed, seed+1, attempts+1
                try:
                    path, source = download_basket(db, cfg, stock_seed, folder)
                    panel = load_panel(path, cfg)
                    checked_range(panel, role, cfg)
                    if used.intersection(panel.permnos) or entities.intersection(panel.entityids):
                        raise ValueError("Stock or issuer overlaps an already accepted basket.")
                    used.update(panel.permnos)
                    entities.update(panel.entityids)
                    files[role].append(path)
                    records.append(dict(role=role, stock_seed=stock_seed, status="accepted", source=source,
                                        panel=str(path.relative_to(cfg.root)), reason=""))
                    print(f"Accepted {role} {len(files[role])}/{needed}, seed {stock_seed} ({source})")
                except ValueError as exc:
                    records.append(dict(role=role, stock_seed=stock_seed, status="rejected", reason=str(exc)))
                    print(f"Rejected seed {stock_seed}: {exc}")
                pd.DataFrame(records).to_csv(folder/"collection_attempts.csv", index=False)
            if len(files[role]) != needed:
                raise ValueError(f"Accepted {len(files[role])}/{needed} {role} baskets after {attempts} attempts. "
                                 f"Inspect {folder/'collection_attempts.csv'}. No dates or scores were silently changed.")
    manifest = dict(source="WRDS", refresh_requested=cfg.refresh_wrds,
        files={role: [str(p.relative_to(cfg.root)) for p in paths] for role, paths in files.items()})
    manifest_path = folder/"data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Local data manifest:", manifest_path)
    return files
