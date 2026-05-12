import argparse
import csv
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise SystemExit("GITHUB_TOKEN not set in .env")

GH = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

CUTOFF = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
TODAY  = datetime.now(timezone.utc).strftime("%Y-%m-%d")

SCHOOLS = [
    "UIUC", "University of Maryland", "Georgia Tech", "University of Michigan",
    "University of Chicago", "Northwestern University", "Rice University",
    "University of Waterloo", "UT Austin", "University of Washington",
    "Cornell University", "Carnegie Mellon", "UC San Diego", "Purdue University",
    "Johns Hopkins", "Harvey Mudd College", "Caltech", "Notre Dame",
    "Texas A&M", "University of Virginia", "Rochester Institute of Technology",
    "University of Toronto", "University of British Columbia",
    "University of Wisconsin Madison",
]

PENALTY_WORDS = {"homework", "assignment", "course", "tutorial", "class", "lecture", "exam"}

CATEGORY_SIGNALS = {
    "AI Infra":      {"inference", "serving", "deployment", "runtime", "scheduler",
                      "llm", "model serving", "inference engine", "triton", "cuda",
                      "throughput", "latency", "quantiz"},
    "ML Systems":    {"training", "distributed", "parallelism", "gradient", "optimizer",
                      "pytorch", "jax", "accelerat", "mixed precision", "checkpoint",
                      "pipeline parallel"},
    "Databases":     {"database", "sql", "query", "storage", "index",
                      "vector db", "embedding", "retrieval", "olap", "oltp", "transaction"},
    "Devtools":      {"compiler", "static analysis", "debugging", "profiling", "linter",
                      "code generation", "synthesis", "verification", "formal",
                      "program analysis", "patch"},
    "Security":      {"vulnerability", "fuzzing", "exploit", "cryptography", "privacy",
                      "malware", "detection", "adversarial", "attack", "defense", "threat"},
    "NLP/Agents":    {"language model", "llm", "agent", "reasoning", "retrieval",
                      "transformer", "fine-tun", "rlhf", "prompt", "chat", "instruct", "rag"},
    "Robotics":      {"robot", "locomotion", "manipulation", "drone", "sim",
                      "navigation", "planning", "control", "actuator", "lidar", "slam",
                      "autonomous"},
    "Bioinformatics": {"genomic", "protein", "dna", "rna", "sequenc", "molecular", "drug"},
}


# ── helpers (unchanged) ───────────────────────────────────────────────────────

def get(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=GH, params=params, timeout=30)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if attempt == 2:
                print(f"  [failed after 3 retries] {url}")
                return {}
            print(f"  [network error] retrying in 10s...")
            time.sleep(10)
            continue
        if r.status_code in (403, 429):
            wait = max(int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time()), 5)
            print(f"rate limit — waiting {wait}s")
            time.sleep(wait)
            continue
        if r.status_code in (404, 422):
            return {} if isinstance(r.json(), dict) else []
        return r.json()
    return {}


PAPER_PLAIN = {"paper", "preprint", "proceedings"}


ARXIV_RE = re.compile(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', re.IGNORECASE)


def extract_arxiv_url(text: str) -> str:
    m = ARXIV_RE.search(text)
    return f"https://arxiv.org/abs/{m.group(1)}" if m else ""


def has_paper_signal(text: str) -> bool:
    tl = text.lower()
    if "arxiv.org" in tl:
        return True
    if any(kw in tl for kw in PAPER_PLAIN):
        return True
    if re.search(r'\[[A-Z]{2,8}[\s\'"]\d{2,4}\]', text):   # [SOSP'25], [ICLR 2026]
        return True
    if re.search(r'\([A-Z]{2,8}[\s\'"]\d{2,4}\)', text):   # (NeurIPS 2025), (OOPSLA 2024)
        return True
    if re.search(r'\([A-Z]{2,8}\d{4}\)', text):             # (VLDB2026), (ICML2025)
        return True
    if re.search(r'\d{4}\)$', text.strip()):                # ends in year: ...Systems (SOSP 2024)
        return True
    return False


def find_orgs(school):
    seen, logins = set(), []
    for query in [f'type:org "{school}"', f'type:org "{school}" in:description']:
        data = get("https://api.github.com/search/users", {"q": query, "per_page": 15})
        time.sleep(2)
        for item in (data.get("items") or []):
            if item["login"] not in seen:
                seen.add(item["login"])
                logins.append(item["login"])
    return logins


def get_repos(org):
    info = get(f"https://api.github.com/orgs/{org}")
    time.sleep(1)
    n = info.get("public_repos", 0) if isinstance(info, dict) else 0
    if n > 300:
        print(f"  skipping {org} (too large: {n} repos)")
        return []
    repos, page = [], 1
    while True:
        batch = get(f"https://api.github.com/orgs/{org}/repos", {"type": "public", "per_page": 100, "page": page})
        if not isinstance(batch, list) or not batch:
            break
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return repos


def edu_emails(org, repo):
    commits = get(f"https://api.github.com/repos/{org}/{repo}/commits", {"per_page": 20})
    if not isinstance(commits, list):
        return []
    return list({
        c["commit"][key]["email"]
        for c in commits
        for key in ("author", "committer")
        if c.get("commit", {}).get(key, {}).get("email", "").endswith(".edu")
    })


def fetch_readme(org, repo):
    r = requests.get(
        f"https://api.github.com/repos/{org}/{repo}/readme",
        headers={**GH, "Accept": "application/vnd.github.v3.raw"},
        timeout=15,
    )
    time.sleep(1)
    return r.text if r.ok else ""


def readme_signals(text):
    has_paper   = has_paper_signal(text)
    has_product = any(kw in text.lower() for kw in ("pip install", "demo", "api endpoint", "docker", "pypi", "package"))
    return has_paper, has_product


def stars_per_month(r):
    created = r.get("created_at") or ""
    if not created:
        return 0
    months = max((datetime.now(timezone.utc) - datetime.fromisoformat(created.replace("Z", "+00:00"))).days / 30, 1)
    return round(r.get("stargazers_count", 0) / months, 2)


def categorize(text):
    for category, signals in CATEGORY_SIGNALS.items():
        if any(s in text for s in signals):
            return category
    return "Uncategorized"


RESEARCH_LANG = {"Python", "C++", "Rust", "Go", "Julia"}
RESEARCH_ORG_SIGNALS = {"lab", "research", "institute", "group", "center", "ai", "ml",
                        "db", "sys", "nlp", "vision", "robot"}


def compute_score(row, has_paper, has_product):
    score = 0
    if has_paper:                                                    score += 3
    if any(s in row["org"].lower() for s in RESEARCH_ORG_SIGNALS):  score += 2
    if has_product:                                                  score += 2
    if row["star_velocity"] > 20:                                    score += 2
    if row["stars"] > 100:                                           score += 2
    if row["forks"] > 20:                                            score += 1
    if row["topics"]:                                                score += 1
    if row["edu_verified"]:                                          score += 1
    if row["language"] in RESEARCH_LANG:                             score += 1
    penalty_text = (row["repo"] + " " + row["description"]).lower()
    if any(w in penalty_text for w in PENALTY_WORDS):
        score -= 3
    return score


# ── digest helpers ────────────────────────────────────────────────────────────

def load_previous_run(runs_dir: Path) -> dict:
    """Return {url: row} from the most recent previous run file, or {} if none."""
    past = sorted(f for f in runs_dir.glob("*.csv") if f.stem != TODAY)
    if not past:
        return {}
    with open(past[-1], newline="", encoding="utf-8") as f:
        return {row["url"]: row for row in csv.DictReader(f)}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ── run ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--schools", type=int, default=None, help="Only run the first N schools")
args = parser.parse_args()

target_schools = SCHOOLS[: args.schools] if args.schools else SCHOOLS

extra_orgs_file = Path("extra_orgs.txt")
extra_orgs = [
    line.strip() for line in extra_orgs_file.read_text().splitlines()
    if line.strip() and not line.startswith("#")
] if extra_orgs_file.exists() else []

extra_org = os.getenv("EXTRA_ORG", "").strip()
if extra_org and extra_org not in extra_orgs:
    extra_orgs.append(extra_org)

Path("runs").mkdir(exist_ok=True)
start = time.time()
rows  = []

# ── Extra orgs (from extra_orgs.txt + optional workflow_dispatch input) ───────
for extra_org in extra_orgs:
    print(f"[extra org] scraping {extra_org} ...")
    for r in get_repos(extra_org):
        if r.get("fork") or r.get("archived"):
            continue
        if (r.get("pushed_at") or "") < CUTOFF:
            continue
        emails = edu_emails(extra_org, r["name"])
        desc_has_paper = has_paper_signal(r.get("description") or "")
        readme = ""
        if r.get("stargazers_count", 0) >= 5:
            readme = fetch_readme(extra_org, r["name"])
            readme_has_paper, has_product = readme_signals(readme)
            has_paper = desc_has_paper or readme_has_paper
        else:
            has_paper, has_product = desc_has_paper, False
        arxiv_url = (extract_arxiv_url(r.get("description") or "") or extract_arxiv_url(readme)) if has_paper else ""
        velocity  = stars_per_month(r)
        topics    = r.get("topics") or []
        tag_text  = " ".join([r.get("name", ""), r.get("description", "") or "", " ".join(topics)]).lower()
        row = {
            "scraped_at":        TODAY,
            "school":            "External",
            "org":               extra_org,
            "repo":              r["name"],
            "url":               r["html_url"],
            "stars":             r["stargazers_count"],
            "forks":             r["forks_count"],
            "last_updated":      (r.get("pushed_at") or "")[:10],
            "description":       (r.get("description") or "")[:120],
            "edu_verified":      bool(emails),
            "edu_emails":        ", ".join(emails),
            "language":          r.get("language") or "",
            "topics":            ", ".join(topics),
            "open_issues":       r.get("open_issues_count", 0),
            "size_kb":           r.get("size", 0),
            "star_velocity":     velocity,
            "has_paper":         has_paper,
            "arxiv_url":         arxiv_url,
            "has_product_signal": has_product,
            "category":          categorize(tag_text),
            "potential_score":   0,
        }
        row["potential_score"] = compute_score(row, has_paper, has_product)
        rows.append(row)

for school_idx, school in enumerate(target_schools, 1):
    elapsed = time.time() - start
    if school_idx > 1:
        eta    = elapsed / school_idx * len(target_schools) - elapsed
        timing = f"{fmt_time(elapsed)} elapsed | ~{fmt_time(eta)} remaining"
    else:
        timing = f"{fmt_time(elapsed)} elapsed"
    print(f"[{timing}] ({school_idx}/{len(target_schools)}) {school}", end=" ... ", flush=True)
    orgs = find_orgs(school)
    print(orgs)

    for org in orgs:
        repos = get_repos(org)
        for r in repos:
            if r.get("fork") or r.get("archived"):
                continue
            if (r.get("pushed_at") or "") < CUTOFF:
                continue

            emails = edu_emails(org, r["name"])
            desc_has_paper = has_paper_signal(r.get("description") or "")

            readme = ""
            if r.get("stargazers_count", 0) >= 5 and bool(emails):
                readme = fetch_readme(org, r["name"])
                readme_has_paper, has_product = readme_signals(readme)
                has_paper = desc_has_paper or readme_has_paper
            else:
                has_paper, has_product = desc_has_paper, False

            if has_paper:
                arxiv_url = extract_arxiv_url(r.get("description") or "") or extract_arxiv_url(readme)
            else:
                arxiv_url = ""

            velocity = stars_per_month(r)
            topics   = r.get("topics") or []
            tag_text = " ".join([r.get("name", ""), r.get("description", "") or "", " ".join(topics)]).lower()
            category = categorize(tag_text)

            row = {
                "scraped_at":         TODAY,
                "school":             school,
                "org":                org,
                "repo":               r["name"],
                "url":                r["html_url"],
                "stars":              r["stargazers_count"],
                "forks":              r["forks_count"],
                "last_updated":       (r.get("pushed_at") or "")[:10],
                "description":        (r.get("description") or "")[:120],
                "edu_verified":       bool(emails),
                "edu_emails":         ", ".join(emails),
                "language":           r.get("language") or "",
                "topics":             ", ".join(topics),
                "open_issues":        r.get("open_issues_count", 0),
                "size_kb":            r.get("size", 0),
                "star_velocity":      velocity,
                "has_paper":          has_paper,
                "arxiv_url":          arxiv_url,
                "has_product_signal": has_product,
                "category":           category,
                "potential_score":    0,
            }
            row["potential_score"] = compute_score(row, has_paper, has_product)
            rows.append(row)

    # Flush after each school so a crash doesn't lose everything
    rows.sort(key=lambda x: -x["potential_score"])
    write_csv(Path("results.csv"), rows)
    write_csv(Path(f"runs/{TODAY}.csv"), rows)

rows.sort(key=lambda x: -x["potential_score"])

# ── save run ──────────────────────────────────────────────────────────────────

write_csv(Path("results.csv"), rows)
write_csv(Path(f"runs/{TODAY}.csv"), rows)

# ── weekly digest ─────────────────────────────────────────────────────────────

prev = load_previous_run(Path("runs"))
today_by_url = {r["url"]: r for r in rows}

digest_rows = []
for row in rows:
    url         = row["url"]
    prev_row    = prev.get(url)
    new_repo    = prev_row is None
    star_delta  = row["stars"] - int(prev_row["stars"] if prev_row else row["stars"])
    score_delta = row["potential_score"] - int(prev_row["potential_score"] if prev_row else row["potential_score"])
    is_hot      = star_delta >= 5
    digest_rows.append({
        **row,
        "new_repo":    new_repo,
        "is_hot":      is_hot,
        "star_delta":  star_delta,
        "score_delta": score_delta,
    })

digest_rows.sort(key=lambda x: (not x["new_repo"], -x["star_delta"], -x["potential_score"]))
write_csv(Path("weekly_digest.csv"), digest_rows)

# ── terminal summary ──────────────────────────────────────────────────────────

new_repos  = [r for r in digest_rows if r["new_repo"]]
hot_repos  = [r for r in digest_rows if r["is_hot"]]

print(f"\n{'='*60}")
print(f"Weekly Digest — {TODAY}")
print(f"  New repos found:  {len(new_repos)}")
print(f"  Repos gone hot:   {len(hot_repos)}  (star_delta >= 20)")

print(f"\nTop 10 new repos by potential_score:")
print(f"  {'Repo':<40} {'School':<22} {'Stars':<7} {'Score'}")
print(f"  {'-'*80}")
for r in sorted(new_repos, key=lambda x: -x["potential_score"])[:10]:
    name = f"{r['org']}/{r['repo']}"
    print(f"  {name:<40} {r['school']:<22} {r['stars']:<7} {r['potential_score']}")

print(f"\nTop 5 hot repos by star_delta:")
print(f"  {'Repo':<40} {'Star Delta':<12} {'Stars'}")
print(f"  {'-'*60}")
for r in sorted(hot_repos, key=lambda x: -x["star_delta"])[:5]:
    name = f"{r['org']}/{r['repo']}"
    print(f"  {name:<40} +{r['star_delta']:<11} {r['stars']}")

print(f"\n{'='*60}")
print(f"  {len(rows)} repos → results.csv + runs/{TODAY}.csv")
print(f"  {len(digest_rows)} digest entries → weekly_digest.csv")
