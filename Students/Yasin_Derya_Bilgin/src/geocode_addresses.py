from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class GeocodeResult:
    lat: float
    lon: float
    display_name: str


def read_nonempty_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def geocode(
    *,
    base_url: str,
    query: str,
    user_agent: str,
    timeout_s: float,
) -> Optional[GeocodeResult]:
    url = (
        base_url.rstrip("/")
        + "/search?"
        + urllib.parse.urlencode({"q": query, "format": "jsonv2", "limit": 1})
    )
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload:
        return None
    hit = payload[0]
    return GeocodeResult(
        lat=float(hit["lat"]),
        lon=float(hit["lon"]),
        display_name=str(hit.get("display_name", "")),
    )


def write_points_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "type", "address", "lat", "lon", "display_name"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adres .txt dosyalarını (depo/müşteri ayrı) Nominatim ile enlem-boylama çevirir."
    )
    parser.add_argument("--depots-file", required=True, help="Depo adresleri .txt")
    parser.add_argument("--customers-file", required=True, help="Müşteri adresleri .txt")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Nominatim base URL (varsayılan: http://localhost:8080)",
    )
    parser.add_argument(
        "--user-agent",
        default="RoutingAlg/1.0 (local nominatim)",
        help="Nominatim için User-Agent",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=50,
        help="İstekler arası bekleme (ms) (varsayılan: 50)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=10.0,
        help="HTTP timeout (s) (varsayılan: 10)",
    )
    parser.add_argument(
        "--cache",
        default="data/geocode_cache.json",
        help="Cache JSON dosyası (varsayılan: data/geocode_cache.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/geocoded",
        help="Çıktı klasörü (varsayılan: data/geocoded)",
    )
    parser.add_argument(
        "--prefix",
        default="dataset",
        help="Çıktı dosya öneki (varsayılan: dataset)",
    )
    args = parser.parse_args()

    depots_path = Path(args.depots_file)
    customers_path = Path(args.customers_file)
    out_dir = Path(args.out_dir)
    cache_path = Path(args.cache)

    depots = read_nonempty_lines(depots_path)
    customers = read_nonempty_lines(customers_path)

    cache = load_cache(cache_path)
    failures: list[str] = []

    def geocode_cached(q: str) -> Optional[GeocodeResult]:
        if q in cache and cache[q].get("ok"):
            return GeocodeResult(
                lat=float(cache[q]["lat"]),
                lon=float(cache[q]["lon"]),
                display_name=str(cache[q].get("display_name", "")),
            )
        if q in cache and not cache[q].get("ok"):
            return None
        try:
            res = geocode(
                base_url=args.base_url,
                query=q,
                user_agent=args.user_agent,
                timeout_s=args.timeout_s,
            )
        except Exception as e:  # noqa: BLE001 - CLI tool, show all failures
            cache[q] = {"ok": False, "error": repr(e)}
            save_cache(cache_path, cache)
            return None

        if res is None:
            cache[q] = {"ok": False}
            save_cache(cache_path, cache)
            return None

        cache[q] = {"ok": True, "lat": res.lat, "lon": res.lon, "display_name": res.display_name}
        save_cache(cache_path, cache)
        return res

    depot_rows: list[dict[str, Any]] = []
    for i, addr in enumerate(depots, start=1):
        res = geocode_cached(addr)
        if res is None:
            failures.append(f"DEPOT\t{i}\t{addr}")
            continue
        depot_rows.append(
            {
                "id": f"D{i}",
                "type": "depot",
                "address": addr,
                "lat": res.lat,
                "lon": res.lon,
                "display_name": res.display_name,
            }
        )
        time.sleep(max(0.0, args.sleep_ms / 1000.0))

    customer_rows: list[dict[str, Any]] = []
    for i, addr in enumerate(customers, start=1):
        res = geocode_cached(addr)
        if res is None:
            failures.append(f"CUSTOMER\t{i}\t{addr}")
            continue
        customer_rows.append(
            {
                "id": f"C{i}",
                "type": "customer",
                "address": addr,
                "lat": res.lat,
                "lon": res.lon,
                "display_name": res.display_name,
            }
        )
        time.sleep(max(0.0, args.sleep_ms / 1000.0))

    depots_out = out_dir / f"{args.prefix}_depots.csv"
    customers_out = out_dir / f"{args.prefix}_customers.csv"
    write_points_csv(depots_out, depot_rows)
    write_points_csv(customers_out, customer_rows)

    if failures:
        failures_path = out_dir / f"{args.prefix}_geocode_failures.txt"
        failures_path.write_text("\n".join(failures).strip() + "\n", encoding="utf-8")
        print(f"Geocode tamamlandı ama {len(failures)} adres çözülemedi: {failures_path}")
    else:
        print("Geocode tamamlandı; tüm adresler çözüldü.")

    print(f"Depo CSV: {depots_out}")
    print(f"Müşteri CSV: {customers_out}")
    print(f"Cache: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

