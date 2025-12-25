from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import math


def load_points(path: Path) -> dict[int, dict[str, float | str]]:
    points: dict[int, dict[str, float | str]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            points[idx] = {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "id": row["id"],
            "type": row["type"],
        }
    return points


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    r = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def compute_route_distance(
    depot_idx: int, stops_idx: list[int], points: dict[int, dict[str, float | str]]
) -> float:
    if not stops_idx:
        return 0.0
    depot = points[depot_idx]
    prev_lat = float(depot["lat"])
    prev_lon = float(depot["lon"])
    total = 0.0
    for idx in stops_idx:
        point = points[idx]
        cur_lat = float(point["lat"])
        cur_lon = float(point["lon"])
        total += haversine_km(prev_lat, prev_lon, cur_lat, cur_lon)
        prev_lat, prev_lon = cur_lat, cur_lon
    total += haversine_km(prev_lat, prev_lon, float(depot["lat"]), float(depot["lon"]))
    return total


def plot_km_chart(
    data: dict, points: dict[int, dict[str, float | str]], out_path: Path, inline: bool
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit("matplotlib bulunamadı. `pip install matplotlib` ile kurun.") from exc

    labels: list[str] = []
    kms: list[float] = []
    minutes: list[float] = []
    customers: list[int] = []

    for depot in data["depots"]:
        depot_id = depot["depot"]["id"]
        depot_idx = depot["depot"]["idx"]
        for route in depot.get("routes", []):
            label = f"{depot_id}-R{route['route_idx']}"
            km = route.get("km")
            if km is None:
                km = compute_route_distance(depot_idx, route.get("stops_idx", []), points)
            labels.append(label)
            kms.append(float(km))
            minutes.append(float(route.get("minutes", 0.0)))
            customers.append(route.get("customers", len(route.get("stops_idx", []))))

    plt.figure(figsize=(5, 3))
    bars = plt.bar(labels, kms, color="#377eb8")
    plt.xlabel("Arac (Depo-Rota)")
    plt.ylabel("Rota km")
    plt.title("Arac bazinda rota uzunluklari")

    for bar, km, cust, minute in zip(bars, kms, customers, minutes):
        text = f"{km:.1f} km"
        if minute > 0:
            text += f"\n{minute:.0f} dk"
        text += f"\n{cust} musteri"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ylim = max(kms) * 1.25 if kms else 1.0
    plt.ylim(0, ylim)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    if inline:
        try:  # pragma: no cover - optional display
            from IPython.display import display

            display(plt.gcf())
        except Exception:
            pass
    plt.close()
    print(f"KM grafiği kaydedildi: {out_path}")


def plot_routes_map(
    data: dict,
    points: dict[int, dict[str, float | str]],
    out_path: Path,
    center_lat: float | None,
    center_lon: float | None,
    zoom: int,
    inline: bool,
) -> None:
    try:
        import folium
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit("folium bulunamadı. `pip install folium` ile kurun.") from exc

    if center_lat is None or center_lon is None:
        depot_coords = []
        for depot in data["depots"]:
            depot_idx = depot["depot"]["idx"]
            depot_point = points[depot_idx]
            depot_coords.append((depot_point["lat"], depot_point["lon"]))
        if depot_coords:
            center_lat = sum(lat for lat, _ in depot_coords) / len(depot_coords)
            center_lon = sum(lon for _, lon in depot_coords) / len(depot_coords)
        else:
            first = next(iter(points.values()), None)
            if not first:
                raise ValueError("Harita çizmek için nokta bulunamadı.")
            center_lat = float(first["lat"])
            center_lon = float(first["lon"])

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

    palette = [
        "#377eb8",
        "#e41a1c",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#999999",
    ]

    def route_colors(route_idx: int) -> tuple[str, str, str]:
        if route_idx == 1:
            return "green", "#377eb8", "red"
        if route_idx == 2:
            return "#40E0D0", "#800080", "#000000"
        mid = palette[(route_idx - 1) % len(palette)]
        return "green", mid, "red"

    total_routes = sum(len(info["routes"]) for info in data["depots"])

    for depot_info in data["depots"]:
        depot_idx = depot_info["depot"]["idx"]
        depot_point = points[depot_idx]
        folium.Marker(
            [depot_point["lat"], depot_point["lon"]],
            popup=f"Depo: {depot_point['id']} (ba?lang??/biti?)",
            icon=folium.Icon(color="red", icon="truck", prefix="fa"),
        ).add_to(m)

        for route_idx, route in enumerate(depot_info["routes"], start=1):
            stops = route["stops_idx"]
            if not stops:
                continue

            start_color, route_color, end_color = route_colors(route_idx)
            if total_routes > 2:
                unified = palette[(depot_idx + route_idx) % len(palette)]
                start_color = route_color = end_color = unified

            def add_arrow_head(start_lat: float, start_lon: float, end_lat: float, end_lon: float, color: str) -> None:
                meters_lat = 111_320.0
                meters_lon = max(111_320.0 * math.cos(math.radians((start_lat + end_lat) / 2.0)), 1e-6)
                dx_m = (end_lon - start_lon) * meters_lon
                dy_m = (end_lat - start_lat) * meters_lat
                norm = math.hypot(dx_m, dy_m)
                if norm == 0:
                    return
                arrow_len = min(max(0.3 * norm, 35.0), 80.0)
                half = arrow_len / 2.0
                if norm < 5.0:  # segment neredeyse nokta
                    return
                frac_shift = half / norm
                center_frac = 0.5
                base_frac = max(0.0, center_frac - frac_shift)
                tip_frac = min(1.0, center_frac + frac_shift)
                base_lat = start_lat + (end_lat - start_lat) * base_frac
                base_lon = start_lon + (end_lon - start_lon) * base_frac
                tip_lat = start_lat + (end_lat - start_lat) * tip_frac
                tip_lon = start_lon + (end_lon - start_lon) * tip_frac
                width_m = min(max(arrow_len * 0.35, 12.0), 40.0)
                perp_x = -dy_m / norm
                perp_y = dx_m / norm
                left_lat = base_lat + (perp_y * width_m) / meters_lat
                left_lon = base_lon + (perp_x * width_m) / meters_lon
                right_lat = base_lat - (perp_y * width_m) / meters_lat
                right_lon = base_lon - (perp_x * width_m) / meters_lon
                folium.Polygon(
                    locations=[
                        [tip_lat, tip_lon],
                        [left_lat, left_lon],
                        [right_lat, right_lon],
                    ],
                    color=color,
                    weight=0,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                ).add_to(m)

            first_point = points[stops[0]]
            start_coords = [
                [depot_point["lat"], depot_point["lon"]],
                [first_point["lat"], first_point["lon"]],
            ]
            folium.PolyLine(
                start_coords,
                color=start_color,
                weight=5,
                opacity=0.9,
                tooltip=f"Depo ? {first_point['id']} (rota {route_idx})",
            ).add_to(m)
            add_arrow_head(
                start_coords[0][0],
                start_coords[0][1],
                start_coords[1][0],
                start_coords[1][1],
                start_color,
            )

            for step_no, cust_idx in enumerate(stops, start=1):
                cust_point = points[cust_idx]
                folium.Marker(
                    [cust_point["lat"], cust_point["lon"]],
                    icon=folium.DivIcon(
                        html=f"<div style='font-size:9px;color:#08306b'><b>{step_no}</b></div>"
                    ),
                    popup=f"{step_no}. durak - {cust_point['id']}",
                ).add_to(m)

            for a_idx, b_idx in zip(stops, stops[1:]):
                a_point = points[a_idx]
                b_point = points[b_idx]
                segment = [
                    [a_point["lat"], a_point["lon"]],
                    [b_point["lat"], b_point["lon"]],
                ]
                folium.PolyLine(
                    segment,
                    color=route_color,
                    weight=4,
                    opacity=0.7,
                    tooltip=f"{points[a_idx]['id']} ? {points[b_idx]['id']}",
                ).add_to(m)
                add_arrow_head(
                    segment[0][0],
                    segment[0][1],
                    segment[1][0],
                    segment[1][1],
                    route_color,
                )

            last_point = points[stops[-1]]
            end_coords = [
                [last_point["lat"], last_point["lon"]],
                [depot_point["lat"], depot_point["lon"]],
            ]
            folium.PolyLine(
                end_coords,
                color=end_color,
                weight=5,
                opacity=0.9,
                tooltip=f"{last_point['id']} ? depo (rota {route_idx} d?n??)",
            ).add_to(m)
            add_arrow_head(
                end_coords[0][0],
                end_coords[0][1],
                end_coords[1][0],
                end_coords[1][1],
                end_color,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(out_path)
    if inline:
        try:  # pragma: no cover
            from IPython.display import display

            display(m)
        except Exception:
            pass
    print(f"Harita kaydedildi: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="run_vrp.py çıktısı (JSON + points CSV) üzerinden grafik ve harita üretir."
    )
    parser.add_argument("--routes-json", required=True, help="run_vrp.py --out ile üretilen JSON")
    parser.add_argument("--points-csv", required=True, help="run_vrp.py --export-points ile kaydedilen CSV")
    parser.add_argument("--map-html", default="out/routes_map.html", help="Folium harita çıktısı")
    parser.add_argument("--km-chart", default="out/routes_km.png", help="Matplotlib km grafiği çıktısı")
    parser.add_argument("--center-lat", type=float, help="Harita merkez lat (opsiyonel)")
    parser.add_argument("--center-lon", type=float, help="Harita merkez lon (opsiyonel)")
    parser.add_argument("--zoom", type=int, default=12, help="Harita zoom (varsayılan 12)")
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Jupyter'de çalışırken harita ve grafiği ekranda da göster (kaydetmeye ek).",
    )
    parser.add_argument("--no-map", action="store_true", help="Harita üretme")
    parser.add_argument("--no-km", action="store_true", help="KM grafiği üretme")
    args = parser.parse_args()

    data = json.loads(Path(args.routes_json).read_text(encoding="utf-8"))
    points = load_points(Path(args.points_csv))

    if not args.no_km:
        plot_km_chart(data, points, Path(args.km_chart), inline=args.inline)
    if not args.no_map:
        plot_routes_map(
            data=data,
            points=points,
            out_path=Path(args.map_html),
            center_lat=args.center_lat,
            center_lon=args.center_lon,
            zoom=args.zoom,
            inline=args.inline,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
