from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------
# Referans tablolar (CLI ciktilari icin)
# ---------------------------------------------------------
FUNCTION_SUMMARY: dict[str, str] = {
    'load_points_from_geocoded': 'Depo ve musteri CSV satirlarini Point nesnelerine cevirir.',
    'write_points_index': 'Point listesini idx/id/lat/lon kolonlariyla CSV\'ye yazar.',
    'build_distance_matrix': 'Haversine tabanli NxN km matrisi olusturur.',
    'build_osrm_distance_matrix': 'OSRM /table endpointinden km ve dakika matrisleri ceker.',
    'split_by_cuts': 'Tek dizide tutulan musterileri kesme noktalarina gore rotalara boler.',
    'two_opt': 'Her rota icin lokal arama ile kisa devreleri duzeltir.',
    'solve_depot_vrp_ga': 'Genetik algoritmanin tum evrimsel adimlarini yurutur.',
    'save_routes_map': 'Folium ile rotalari HTML harita uzerine cizer (opsiyonel).',
    'save_depot_km_chart': 'Matplotlib bar grafigi ile depo bazli km/dk raporu uretir.',
    'parse_vehicles_per_depot': 'CLI\'da girilen sayi listelerini tamsayiya donusturur.',
    'main': 'Tum CLI kontrol akisini yoneten ana giris noktasi.',
}

GA_PARAM_DEFAULTS = {
    'pop_size': 60,
    'generations': 250,
    'seed': 42,
    'crossover_p': 0.9,
    'mutation_p': 0.25,
    'cut_mutation_p': 0.20,
    'load_balance_weight': 1.0,
    'objective': 'distance',
    'duration_weight': 1.0,
}

GA_PARAM_DOCS = {
    'pop_size': 'Genetik populasyondaki birey sayisi; cesitliligi belirler.',
    'generations': 'Evrim dongusu kac kez calistirilacak?',
    'seed': 'Rastgelelik tekrarlanabilir olsun diye kullanilan tohum.',
    'crossover_p': 'Ordered crossover\'in uygulanma olasiligi.',
    'mutation_p': 'Swap mutasyonunun uygulanma olasiligi.',
    'cut_mutation_p': 'Kesme noktalarini yeniden uretme olasiligi.',
    'load_balance_weight': 'Araclar arasindaki musteri sayisi farkini penalize eder.',
    'objective': 'distance/duration/weighted secimi fitness fonksiyonunu belirler.',
    'duration_weight': 'weighted modunda dakikanin km\'ye gore agirligi.',
}

GITHUB_ORIGIN_NOTE = 'Bu dosyadan tespit edemiyorum; kanit yok.'

############################################################
# Point
# Amac: Depo veya musteri koordinatini temsil eden veri sinifi.
# Girdi/Cikti: idx/id/type/lat/lon/address alanlarini saklar.
# Onemli detaylar: GA hesaplamalarinda indeks referansi olarak kullanilir.
############################################################
@dataclass(frozen=True)
class Point:
    idx: int
    id: str
    type: str  # depot | customer
    lat: float
    lon: float
    address: str

############################################################
# load_points_from_geocoded
# Amac: Geocode edilmis CSV'leri Point listesine cevirmek.
# Girdi/Cikti: depots_csv Path, customers_csv Path -> List[Point]
# Onemli detaylar: IDX sirasi korunur; depo once musteriler sonra eklenir.
############################################################
def load_points_from_geocoded(depots_csv: Path, customers_csv: Path) -> list[Point]:
    points: list[Point] = []
    idx = 0

    def append_from_csv(path: Path, point_type: str, prefix: str) -> None:
        nonlocal idx
        with path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                try:
                    lat = float(row['lat'])
                    lon = float(row['lon'])
                except KeyError as exc:
                    raise ValueError(f"{path} dosyasinda lat/lon alan bulunamadi.") from exc
                point_id = row.get('id') or f"{prefix}{row_idx}"
                points.append(
                    Point(
                        idx=idx,
                        id=point_id,
                        type=point_type,
                        lat=lat,
                        lon=lon,
                        address=row.get('address', ''),
                    )
                )
                idx += 1

    append_from_csv(depots_csv, 'depot', 'D')
    append_from_csv(customers_csv, 'customer', 'C')
    return points

############################################################
# write_points_index
# Amac: Point listesini tekrar CSV'ye yazmak (dokumantasyon icin).
# Girdi/Cikti: path Path, points List[Point] -> None
# Onemli detaylar: --export-points kullanildiginda cagirilir.
############################################################
def write_points_index(path: Path, points: list[Point]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'id', 'type', 'lat', 'lon', 'address'])
        writer.writeheader()
        for p in points:
            writer.writerow(
                {
                    'idx': p.idx,
                    'id': p.id,
                    'type': p.type,
                    'lat': p.lat,
                    'lon': p.lon,
                    'address': p.address,
                }
            )

############################################################
# haversine_km
# Amac: Kus ucusu mesafeyi kilometre cinsinden hesaplamak.
# Girdi/Cikti: lat1/lon1/lat2/lon2 -> float km
# Onemli detaylar: Haversine formulune gore; fallback hesaplamalarda kullanilir.
############################################################
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

############################################################
# build_distance_matrix
# Amac: Tum nokta ciftleri arasindaki kus ucusu mesafeyi hesaplamak.
# Girdi/Cikti: points List[Point] -> matrix List[List[float]]
# Onemli detaylar: OSRM devrede degilse zorunlu olarak kullanilir.
############################################################
def build_distance_matrix(points: list[Point]) -> list[list[float]]:
    n = len(points)
    matrix: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            matrix[i][j] = 0.0 if i == j else haversine_km(a.lat, a.lon, b.lat, b.lon)
    return matrix

############################################################
# duration_matrix_from_distance
# Amac: Km matrisini ortalama hiz parametresine gore dakika matrisine cevirir.
# Girdi/Cikti: distance_matrix, avg_speed_kmh -> duration_matrix
# Onemli detaylar: Cok kucuk hizlarda sayisal tasma olmamasi icin min deger kullanilir.
############################################################
def duration_matrix_from_distance(distance_matrix: list[list[float]], avg_speed_kmh: float) -> list[list[float]]:
    speed = max(avg_speed_kmh, 1e-3)
    factor = 60.0 / speed
    return [[val * factor for val in row] for row in distance_matrix]

############################################################
# build_osrm_distance_matrix
# Amac: OSRM /table endpointinden hem km hem dk matrisleri almak.
# Girdi/Cikti: points list + baglanti ayarlari -> (distance_matrix, duration_matrix)
# Onemli detaylar: chunk_size'lar OSRM konfigundaki max table boyutunu asmamalidir.
############################################################
def build_osrm_distance_matrix(
    points: list[Point],
    *,
    base_url: str,
    profile: str = 'driving',
    chunk_size: int = 80,
    timeout: float = 60.0,
) -> tuple[list[list[float]], list[list[float]]]:
    try:
        import requests
    except ImportError as exc:
        raise SystemExit('requests modulunu kurun: pip install requests') from exc

    if chunk_size < 1:
        raise ValueError('chunk_size en az 1 olmalidir.')

    base_url = base_url.rstrip('/')
    n = len(points)
    dist_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    dur_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    def batched_indices(total: int) -> Iterable[tuple[int, int]]:
        start = 0
        while start < total:
            end = min(total, start + chunk_size)
            yield start, end
            start = end

    for src_start, src_end in batched_indices(n):
        src_slice = list(range(src_start, src_end))
        src_points = [points[i] for i in src_slice]
        for dst_start, dst_end in batched_indices(n):
            dst_slice = list(range(dst_start, dst_end))
            dst_points = [points[i] for i in dst_slice]
            coords_block = src_points + dst_points
            coords_str = ';'.join(f"{p.lon:.8f},{p.lat:.8f}" for p in coords_block)
            url = f"{base_url}/table/v1/{profile}/{coords_str}"
            params = {
                'sources': ';'.join(str(i) for i in range(len(src_points))),
                'destinations': ';'.join(str(len(src_points) + i) for i in range(len(dst_points))),
                'annotations': 'distance,duration',
            }
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"OSRM istegi basarisiz ({resp.status_code}): {resp.text}")
            payload = resp.json()
            distances = payload.get('distances')
            durations = payload.get('durations')
            if distances is None or durations is None:
                raise RuntimeError('OSRM yanitinda distance/duration alanlari eksik.')
            for i, row in enumerate(distances):
                for j, value in enumerate(row):
                    dist_matrix[src_slice[i]][dst_slice[j]] = float('inf') if value is None else float(value) / 1000.0
            for i, row in enumerate(durations):
                for j, value in enumerate(row):
                    dur_matrix[src_slice[i]][dst_slice[j]] = float('inf') if value is None else float(value) / 60.0
    return dist_matrix, dur_matrix

############################################################
# write_distance_matrix
# Amac: Mesafe matrisini CSV olarak kaydetmek.
# Girdi/Cikti: path Path, matrix -> None
# Onemli detaylar: ilk kolon idx, sonrakiler ondalik formatinda saklanir.
############################################################
def write_distance_matrix(path: Path, matrix: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        n = len(matrix)
        writer.writerow(['idx'] + [str(i) for i in range(n)])
        for i, row in enumerate(matrix):
            writer.writerow([str(i)] + [f"{val:.6f}" for val in row])

############################################################
# read_points_index
# Amac: Daha once kaydedilen points.csv dosyasini Point listesine cevirmek.
# Girdi/Cikti: path Path -> List[Point]
# Onemli detaylar: idx siralamasina gore tekrar siralanir.
############################################################
def read_points_index(path: Path) -> list[Point]:
    points: list[Point] = []
    with path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(
                Point(
                    idx=int(row['idx']),
                    id=row['id'],
                    type=row['type'],
                    lat=float(row['lat']),
                    lon=float(row['lon']),
                    address=row.get('address', ''),
                )
            )
    points.sort(key=lambda p: p.idx)
    return points

############################################################
# read_matrix
# Amac: Kaydedilmis mesafe matrisini yeniden liste haline getirmek.
# Girdi/Cikti: path Path -> matrix
# Onemli detaylar: Ilk kolon idx olmalidir, aksi halde hata verilir.
############################################################
def read_matrix(path: Path) -> list[list[float]]:
    matrix: list[list[float]] = []
    with path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or header[0] != 'idx':
            raise ValueError("Matrix CSV format bekleniyor: ilk hucre 'idx'")
        for row in reader:
            matrix.append([float(x) for x in row[1:]])
    return matrix
############################################################
# split_by_cuts
# Amac: Musteri dizisini kesme noktalarina gore rotalara bolmek.
# Girdi/Cikti: order list, cuts list -> List[List[int]]
# Onemli detaylar: Bos parcalar filtrelenir.
############################################################
############################################################
# split_by_cuts
# Amac: Tek sirada tutulan musterileri kesme noktalarina gore rotalara ayirmak.
# Girdi/Cikti: order list, cuts list -> List[List[int]]
# Onemli detaylar: Bos parcalar filtrelenir.
############################################################
def split_by_cuts(order: list[int], cuts: list[int]) -> list[list[int]]:
    if not order:
        return []
    if not cuts:
        return [order]
    parts: list[list[int]] = []
    start = 0
    for c in cuts:
        parts.append(order[start:c])
        start = c
    parts.append(order[start:])
    return [p for p in parts if p]

############################################################
# route_metric_total
# Amac: Verilen matrisle bir rotanin depo-basla-bitis maliyetini hesaplamak.
# Girdi/Cikti: depot_idx, route list, matrix -> float
# Onemli detaylar: Hem km hem dk icin ayni fonksiyon kullanilir.
############################################################
############################################################
# route_metric_total
# Amac: Bir rota icin depo-basla-bitis toplam maliyetini hesaplamak.
# Girdi/Cikti: depot_idx, route list, matrix -> float
# Onemli detaylar: Hem km hem dk hesaplamalari icin kullanilir.
############################################################
def route_metric_total(depot_idx: int, route: list[int], matrix: list[list[float]] | None) -> float:
    if matrix is None or not route:
        return 0.0
    total = matrix[depot_idx][route[0]]
    for a, b in zip(route, route[1:]):
        total += matrix[a][b]
    total += matrix[route[-1]][depot_idx]
    return total

############################################################
# route_distance_km / solution_distance_km
# Amac: Tek rota veya tum cozum icin toplam km'yi hesaplamak.
# Girdi/Cikti: depot_idx, route(s), dist matrix -> float
# Onemli detaylar: Sahne arkasinda route_metric_total kullanilir.
############################################################
############################################################
# route_distance_km
# Amac: Tek bir rotanin kilometre bazli uzunlugunu hesaplamak.
# Girdi/Cikti: depot_idx, route list, dist matrix -> float
# Onemli detaylar: route_metric_total'u km matrisiyle cagirir.
############################################################
def route_distance_km(depot_idx: int, route: list[int], dist: list[list[float]]) -> float:
    return route_metric_total(depot_idx, route, dist)


############################################################
# solution_distance_km
# Amac: Tum rotalarin toplam kilometre uzunlugunu hesaplamak.
# Girdi/Cikti: depot_idx, routes listesi, dist matrix -> float
############################################################
def solution_distance_km(depot_idx: int, routes: list[list[int]], dist: list[list[float]]) -> float:
    return sum(route_distance_km(depot_idx, r, dist) for r in routes)

############################################################
# route_duration_minutes / solution_duration_minutes
# Amac: Sure matrisine gore dakikayi hesaplamak.
# Girdi/Cikti: depot_idx, route(s), duration matrix -> float
# Onemli detaylar: duration_matrix None ise 0 dondurur.
############################################################
############################################################
# route_duration_minutes
# Amac: Tek rotanin dakika bazli uzunlugunu hesaplamak.
# Girdi/Cikti: depot_idx, route list, duration matrix -> float
# Onemli detaylar: duration_matrix None ise 0 dondurur.
############################################################
def route_duration_minutes(depot_idx: int, route: list[int], duration: list[list[float]] | None) -> float:
    return route_metric_total(depot_idx, route, duration)


############################################################
# solution_duration_minutes
# Amac: Tum rotalarin toplam dakika uzunlugunu hesaplamak.
# Girdi/Cikti: depot_idx, routes listesi, duration matrix -> float
############################################################
def solution_duration_minutes(
    depot_idx: int, routes: list[list[int]], duration: list[list[float]] | None
) -> float:
    return sum(route_duration_minutes(depot_idx, r, duration) for r in routes)

############################################################
# nearest_neighbor_order
# Amac: Baslangic bireyi icin greedy siralama olusturmak.
# Girdi/Cikti: depot_idx, customers, dist -> order list
# Onemli detaylar: Her adimda en yakin musteri secilir.
############################################################
############################################################
# nearest_neighbor_order
# Amac: Greedy yaklasimla baslangic bireyi icin musteri sirasi olusturmak.
# Girdi/Cikti: depot_idx, customers list, dist matrix -> order list
# Onemli detaylar: Her adimda depodan veya son musteriden en yakin musteri secilir.
############################################################
def nearest_neighbor_order(depot_idx: int, customers: list[int], dist: list[list[float]]) -> list[int]:
    remaining = set(customers)
    order: list[int] = []
    current = depot_idx
    while remaining:
        nxt = min(remaining, key=lambda j: dist[current][j])
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return order

############################################################
# ordered_crossover
# Amac: Klasik OX operatoruyle ebeveyn dizilerinden cocuk uretmek.
# Girdi/Cikti: a list, b list, rng -> child list
# Onemli detaylar: Musterilerin goreceli sirasi korunur.
############################################################
############################################################
# ordered_crossover
# Amac: Klasik Ordered Crossover (OX) ile iki ebeveynden cocuk sirasi olusturmak.
# Girdi/Cikti: ebeveyn listeleri, rastgele generator -> cocuk listesi
# Onemli detaylar: Musterilerin goreceli sirasi korunur.
############################################################
def ordered_crossover(a: list[int], b: list[int], rng: random.Random) -> list[int]:
    n = len(a)
    if n < 2:
        return a[:]
    i = rng.randrange(n)
    j = rng.randrange(n)
    lo, hi = (i, j) if i < j else (j, i)
    child = [None] * n  # type: ignore[list-item]
    child[lo:hi] = a[lo:hi]
    used = set(child[lo:hi])
    fill = [x for x in b if x not in used]
    k = 0
    for idx in range(n):
        if child[idx] is None:
            child[idx] = fill[k]
            k += 1
    return child  # type: ignore[return-value]

############################################################
# mutate_swap
# Amac: Bireyin rotasinda iki musterinin yerini degistirerek cesitlilik yaratmak.
# Girdi/Cikti: order list, rng -> None
# Onemli detaylar: N < 2 ise degisiklik yapilmaz.
############################################################
############################################################
# mutate_swap
# Amac: Bireyin rotasinda iki musterinin yerini degistirerek cesitlilik saglamak.
# Girdi/Cikti: order list, RNG -> None
# Onemli detaylar: n<2 ise degisiklik yapilmaz.
############################################################
def mutate_swap(order: list[int], rng: random.Random) -> None:
    n = len(order)
    if n < 2:
        return
    i = rng.randrange(n)
    j = rng.randrange(n)
    order[i], order[j] = order[j], order[i]

############################################################
# random_cuts
# Amac: Musteri dizisini rastgele kesme noktalarina gore parcaya ayirmak.
# Girdi/Cikti: n_customers, vehicles, rng -> cuts list
# Onemli detaylar: Kesme sayisi (vehicles - 1) kadar olur.
############################################################
############################################################
# random_cuts
# Amac: Musteri siralarini arac sayisi kadar parcaya ayirmak icin rastgele kesmeler uretmek.
# Girdi/Cikti: musteri sayisi, arac sayisi, RNG -> cuts list
# Onemli detaylar: kesme sayisi vehicles-1 kadardir.
############################################################
def random_cuts(n_customers: int, vehicles: int, rng: random.Random) -> list[int]:
    vehicles = max(1, vehicles)
    if n_customers <= 1 or vehicles <= 1:
        return []
    vehicles = min(vehicles, n_customers)
    return sorted(rng.sample(range(1, n_customers), k=vehicles - 1))

############################################################
# even_cuts
# Amac: Musterileri araclara olabildigince esit dagitmak icin kesmeler uretmek.
# Girdi/Cikti: n_customers, vehicles -> cuts list
# Onemli detaylar: Arta kalan musteriler ilk araclara paylastirilir.
############################################################
############################################################
# even_cuts
# Amac: Musterileri araclar arasinda dengeli dagitacak kesme noktalarini hesaplamak.
# Girdi/Cikti: musteri sayisi, arac sayisi -> cuts list
############################################################
def even_cuts(n_customers: int, vehicles: int) -> list[int]:
    vehicles = max(1, min(vehicles, n_customers))
    if vehicles <= 1:
        return []
    base = n_customers // vehicles
    rem = n_customers % vehicles
    cuts: list[int] = []
    acc = 0
    for i in range(vehicles - 1):
        acc += base + (1 if i < rem else 0)
        cuts.append(acc)
    return cuts

############################################################
# two_opt
# Amac: Rota icindeki kisa devreleri kesip daha iyi siralama bulmak.
# Girdi/Cikti: route list, dist matrix -> optimized list
# Onemli detaylar: 2-opt artik iyilesme olmayana kadar devam eder.
############################################################
############################################################
# two_opt
# Amac: Rota icinde 2-opt lokal aramasi yaparak kisaltma saglamak.
# Girdi/Cikti: route list, dist matrix -> optimize rota
# Onemli detaylar: Iyilesme olmayana kadar 2-opt uygulanir.
############################################################
def two_opt(route: list[int], dist: list[list[float]]) -> list[int]:
    n = len(route)
    if n < 4:
        return route
    improved = True
    best = route[:]
    while improved:
        improved = False
        for i in range(n - 3):
            a = best[i]
            b = best[i + 1]
            for j in range(i + 2, n - 1):
                c = best[j]
                d = best[j + 1]
                gain = (dist[a][b] + dist[c][d]) - (dist[a][c] + dist[b][d])
                if gain > 1e-9:
                    best = best[: i + 1] + list(reversed(best[i + 1 : j + 1])) + best[j + 1 :]
                    improved = True
                    break
            if improved:
                break
    return best
############################################################
# solve_depot_vrp_ga
# Amac: Belirli bir depo icin genetik algoritmayla musterileri araclara dagitmak.
# Girdi/Cikti: GA parametreleri + mesafe/sure matrisleri -> (routes, km, dakika)
# Onemli detaylar: Birey = (order, cuts); fitness objective parametresine gore hesaplanir.
############################################################
def solve_depot_vrp_ga(
    *,
    depot_idx: int,
    customers: list[int],
    vehicles: int,
    dist: list[list[float]],
    duration_matrix: list[list[float]] | None,
    rng: random.Random,
    pop_size: int,
    generations: int,
    crossover_p: float,
    mutation_p: float,
    cut_mutation_p: float,
    load_balance_weight: float,
    objective: str,
    duration_weight: float,
) -> tuple[list[list[int]], float, float]:
    customers = customers[:]
    n = len(customers)
    if n == 0:
        return [], 0.0, 0.0
    vehicles = max(1, min(vehicles, n))

    @dataclass
    class Individual:
        order: list[int]
        cuts: list[int]
        fitness: float

    def evaluate(order: list[int], cuts: list[int]) -> float:
        # order: musterilerin ziyaret sirasi, cuts: bu siralari arac sayisina gore boler.
        routes = split_by_cuts(order, cuts)
        km = solution_distance_km(depot_idx, routes, dist)
        minutes = solution_duration_minutes(depot_idx, routes, duration_matrix)
        metric = km
        if objective == 'duration':
            metric = minutes if duration_matrix is not None else km
        elif objective == 'weighted':
            metric = km + (duration_weight * minutes if duration_matrix is not None else 0.0)
        if load_balance_weight > 0 and vehicles > 0:
            target = n / vehicles
            imbalance = sum(abs(len(r) - target) for r in routes)
            metric += imbalance * load_balance_weight
        return metric

    def tournament(pop: list[Individual], k: int = 3) -> Individual:
        # Turnuva secimi: rastgele k birey al, fitness'i en dusuk olani sec.
        cand = rng.sample(pop, k=min(k, len(pop)))
        return min(cand, key=lambda ind: ind.fitness)

    seed_order = nearest_neighbor_order(depot_idx, customers, dist)
    seed_cuts = even_cuts(n, vehicles)

    population: list[Individual] = []
    population.append(Individual(order=seed_order, cuts=seed_cuts, fitness=evaluate(seed_order, seed_cuts)))
    while len(population) < pop_size:
        shuffled = customers[:]
        rng.shuffle(shuffled)
        cuts = random_cuts(n, vehicles, rng)
        population.append(Individual(order=shuffled, cuts=cuts, fitness=evaluate(shuffled, cuts)))

    best = min(population, key=lambda ind: ind.fitness)

    for _ in range(generations):
        population.sort(key=lambda ind: ind.fitness)
        # Elitizm: en iyi bireyleri dogrudan sonraki nesle kopyala.
        next_pop: list[Individual] = [population[0]]
        if len(population) > 1:
            next_pop.append(population[1])

        while len(next_pop) < pop_size:
            parent1 = tournament(population)
            parent2 = tournament(population)
            child_order = parent1.order[:]
            child_cuts = parent1.cuts[:]

            if rng.random() < crossover_p:
                # Ordered crossover: ebeveynlerin alt dizilerini kesip cocukta birlestirir.
                child_order = ordered_crossover(parent1.order, parent2.order, rng)
                child_cuts = parent1.cuts[:] if rng.random() < 0.5 else parent2.cuts[:]
                if len(child_cuts) != vehicles - 1:
                    child_cuts = even_cuts(n, vehicles)

            if rng.random() < mutation_p:
                mutate_swap(child_order, rng)
            if rng.random() < cut_mutation_p:
                child_cuts = random_cuts(n, vehicles, rng)

            child_fit = evaluate(child_order, child_cuts)
            next_pop.append(Individual(order=child_order, cuts=child_cuts, fitness=child_fit))

        population = next_pop
        generation_best = min(population, key=lambda ind: ind.fitness)
        if generation_best.fitness < best.fitness:
            best = generation_best

    best_routes = split_by_cuts(best.order, best.cuts)
    improved_routes = [two_opt(r, dist) for r in best_routes]
    best_distance = solution_distance_km(depot_idx, improved_routes, dist)
    best_duration = solution_duration_minutes(depot_idx, improved_routes, duration_matrix)
    return improved_routes, best_distance, best_duration
############################################################
# save_routes_map
# Amac: Folium ile rotalari HTML harita uzerine cizmek.
# Girdi/Cikti: points list, results dict, out_path Path -> None
# Onemli detaylar: Opsiyonel; folium yoksa kurulum uyarisi verir.
############################################################
def save_routes_map(points: list[Point], results: dict[str, Any], out_path: Path) -> None:
    try:
        import folium
    except ImportError as exc:
        raise SystemExit('Folium bulunamadi; `pip install folium` ile kurun.') from exc

    points_by_idx = {p.idx: p for p in points}
    depot_coords = [
        (points_by_idx[depot_info['depot']['idx']].lat, points_by_idx[depot_info['depot']['idx']].lon)
        for depot_info in results['depots']
    ]
    if depot_coords:
        center_lat = sum(lat for lat, _ in depot_coords) / len(depot_coords)
        center_lon = sum(lon for _, lon in depot_coords) / len(depot_coords)
    elif points:
        center_lat, center_lon = points[0].lat, points[0].lon
    else:
        raise ValueError('Harita cizmek icin nokta bulunamadi.')

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']

    for depot_idx, depot_info in enumerate(results['depots']):
        depot_point = points_by_idx[depot_info['depot']['idx']]
        folium.Marker(
            location=[depot_point.lat, depot_point.lon],
            popup=f"Depo: {depot_point.id}",
            icon=folium.Icon(color='red', icon='truck', prefix='fa'),
        ).add_to(m)
        for route_idx, route in enumerate(depot_info['routes'], start=1):
            coords = [[depot_point.lat, depot_point.lon]]
            for stop_idx in route['stops_idx']:
                stop_point = points_by_idx[stop_idx]
                coords.append([stop_point.lat, stop_point.lon])
            coords.append([depot_point.lat, depot_point.lon])
            color = palette[(depot_idx + route_idx) % len(palette)]
            folium.PolyLine(coords, color=color, weight=4, opacity=0.8, tooltip=f"{depot_point.id} rota {route_idx}").add_to(m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(out_path)
    print(f"Harita kaydedildi: {out_path}")

############################################################
# save_depot_km_chart
# Amac: Matplotlib ile depo bazli km/dk bar grafigi uretmek.
# Girdi/Cikti: results dict, out_path Path -> None
# Onemli detaylar: matplotlib yoksa uyarir.
############################################################
def save_depot_km_chart(results: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('matplotlib bulunamadi; `pip install matplotlib` ile kurun.') from exc

    labels = []
    kms = []
    minutes = []
    for depot_info in results['depots']:
        labels.append(depot_info['depot']['id'])
        kms.append(depot_info['km'])
        minutes.append(depot_info.get('minutes', 0.0))

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, kms, color='#377eb8')
    ax.set_ylabel('Toplam km')
    ax.set_title('Depo bazinda rota uzunluklari')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for bar, km, minute in zip(bars, kms, minutes):
        text = f"{km:.1f} km"
        if minute:
            text += f" / {minute:.0f} dk"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"KM grafigi kaydedildi: {out_path}")

############################################################
# parse_vehicles_per_depot
# Amac: CLI'dan gelen '4,6' gibi listeleri tamsayi dizisine cevirmek.
# Girdi/Cikti: arg string -> List[int]
# Onemli detaylar: Bos string gelirse bos liste dondurur.
############################################################
def parse_vehicles_per_depot(arg: str) -> list[int]:
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    if not parts:
        return []
    return [int(p) for p in parts]

############################################################
# print_function_reference
# Amac: --explain bayragi icin fonksiyon ve GA ozeti yazdirmak.
# Girdi/Cikti: None -> None
# Onemli detaylar: Tum fonksiyonlar ve GA parametreleri listelenir.
############################################################
def print_function_reference() -> None:
    print('Fonksiyon ozeti:')
    for name, desc in FUNCTION_SUMMARY.items():
        print(f"- {name}: {desc}")
    print('\nGA Parametreleri:')
    for key, desc in GA_PARAM_DOCS.items():
        default = GA_PARAM_DEFAULTS.get(key, 'bilinmiyor')
        print(f"  {key} (default {default}): {desc}")
    print(f"\nKaynak bilgisi: {GITHUB_ORIGIN_NOTE}")

############################################################
# print_ga_params_only
# Amac: --print-ga-params cikisi icin yalnizca GA parametrelerini yazmak.
# Girdi/Cikti: None -> None
# Onemli detaylar: Default degerler kullanilir.
############################################################
def print_ga_params_only() -> None:
    print('GA Parametreleri (varsayilan degerler):')
    for key, default in GA_PARAM_DEFAULTS.items():
        print(f"- {key}: {default}")

############################################################
# main
# Amac: CLI argumanlarini okuyup VRP cozumunu tetiklemek.
# Girdi/Cikti: None -> exit code int
# Onemli detaylar: --explain ve --print-ga-params bayraklarini destekler.
############################################################
def main() -> int:
    parser = argparse.ArgumentParser(description='Genetik algoritma tabanli coklu arac VRP cozucu.')
    parser.add_argument('--points-csv', help='Hazir point CSV (matrix ile birlikte kullanilir).')
    parser.add_argument('--matrix-csv', help='Hazir mesafe matrisi CSV (points ile birlikte kullanilir).')
    parser.add_argument('--depots-csv', help='Geocode edilmis depo CSV (scripts/geocode_addresses.py).')
    parser.add_argument('--customers-csv', help='Geocode edilmis musteri CSV.')
    parser.add_argument('--export-points', help='Point listesini CSV olarak disari ver.')
    parser.add_argument('--export-matrix', help='Mesafe matrisini CSV olarak disari ver.')
    parser.add_argument('--vehicles', type=int, default=8, help='Toplam arac sayisi (tek depo).')
    parser.add_argument('--vehicles-per-depot', default='', help='Coklu depo icin arac sayilari or: 4,6.')
    parser.add_argument('--no-cluster', action='store_true', help='Coklu depo olsa bile tum musterileri ilk depoya bagla.')
    parser.add_argument('--pop-size', type=int, default=GA_PARAM_DEFAULTS['pop_size'], help='Populasyon buyuklugu.')
    parser.add_argument('--generations', type=int, default=GA_PARAM_DEFAULTS['generations'], help='Nesil sayisi.')
    parser.add_argument('--seed', type=int, default=GA_PARAM_DEFAULTS['seed'], help='Rastgele seed.')
    parser.add_argument('--crossover-p', type=float, default=GA_PARAM_DEFAULTS['crossover_p'], help='Crossover olasiligi.')
    parser.add_argument('--mutation-p', type=float, default=GA_PARAM_DEFAULTS['mutation_p'], help='Swap mutasyon olasiligi.')
    parser.add_argument('--cut-mutation-p', type=float, default=GA_PARAM_DEFAULTS['cut_mutation_p'], help='Kesme mutasyonu olasiligi.')
    parser.add_argument('--out', default='out/routes.json', help='Cikti JSON yolu.')
    parser.add_argument('--load-balance-weight', type=float, default=GA_PARAM_DEFAULTS['load_balance_weight'], help='Musteri dengesi cezasi.')
    parser.add_argument('--plot-html', help='Folium harita HTML cikti yolu.')
    parser.add_argument('--plot-km-chart', help='Matplotlib km/dk grafi cikti yolu.')
    parser.add_argument('--distance-mode', choices=['haversine', 'osrm'], default='haversine', help='Mesafe kaynagi.')
    parser.add_argument('--osrm-base-url', default='http://localhost:5000', help='OSRM base URL.')
    parser.add_argument('--osrm-profile', default='driving', help='OSRM profil adi (table icin).')
    parser.add_argument('--osrm-chunk-size', type=int, default=80, help='OSRM table istegindeki blok boyutu.')
    parser.add_argument('--osrm-timeout', type=float, default=60.0, help='OSRM istegi timeout (saniye).')
    parser.add_argument('--objective', choices=['distance', 'duration', 'weighted'], default=GA_PARAM_DEFAULTS['objective'], help='Fitness hedefi.')
    parser.add_argument('--duration-weight', type=float, default=GA_PARAM_DEFAULTS['duration_weight'], help='Weighted modunda dakika agirligi.')
    parser.add_argument('--avg-speed-kmh', type=float, default=40.0, help='Haversine modunda ortalama hiz (km/s).')
    parser.add_argument('--explain', action='store_true', help='Fonksiyon ve GA aciklamalarini yazdir ve cik.')
    parser.add_argument('--print-ga-params', action='store_true', help='Yalnizca GA parametrelerini yazdir ve cik.')

    args = parser.parse_args()

    if args.explain:
        print_function_reference()
        return 0
    if args.print_ga_params:
        print_ga_params_only()
        return 0

    use_ready_matrix = args.points_csv and args.matrix_csv
    use_geocode_csv = args.depots_csv and args.customers_csv
    duration_matrix: list[list[float]] | None = None

    if use_ready_matrix:
        points = read_points_index(Path(args.points_csv))
        dist = read_matrix(Path(args.matrix_csv))
        duration_matrix = duration_matrix_from_distance(dist, args.avg_speed_kmh)
    elif use_geocode_csv:
        points = load_points_from_geocoded(Path(args.depots_csv), Path(args.customers_csv))
        if args.distance_mode == 'osrm':
            dist, duration_matrix = build_osrm_distance_matrix(
                points,
                base_url=args.osrm_base_url,
                profile=args.osrm_profile,
                chunk_size=args.osrm_chunk_size,
                timeout=args.osrm_timeout,
            )
        else:
            dist = build_distance_matrix(points)
            duration_matrix = duration_matrix_from_distance(dist, args.avg_speed_kmh)
        if args.export_points:
            write_points_index(Path(args.export_points), points)
        if args.export_matrix:
            write_distance_matrix(Path(args.export_matrix), dist)
    else:
        parser.error('Ya --points-csv + --matrix-csv ya da --depots-csv + --customers-csv parametrelerini verin.')

    rng = random.Random(args.seed)
    if len(dist) != len(points):
        raise ValueError('Matrix boyutu points listesiyle uyumlu degil.')

    depot_indices = [p.idx for p in points if p.type == 'depot']
    customer_indices = [p.idx for p in points if p.type == 'customer']
    if not depot_indices:
        raise ValueError('En az bir depo gerekli.')

    clusters: dict[int, list[int]] = {d: [] for d in depot_indices}
    if len(depot_indices) == 1 or args.no_cluster:
        clusters[depot_indices[0]] = customer_indices[:]
    else:
        for cust in customer_indices:
            best_depot = min(depot_indices, key=lambda d: dist[d][cust])
            clusters[best_depot].append(cust)

    vehicles_per_depot = parse_vehicles_per_depot(args.vehicles_per_depot)
    if vehicles_per_depot and len(vehicles_per_depot) != len(depot_indices):
        raise ValueError('vehicles-per-depot ile depo sayisi uyusmuyor.')

    results: dict[str, Any] = {'total_km': 0.0, 'total_minutes': 0.0, 'depots': []}

    for idx, depot_idx in enumerate(depot_indices):
        depot_point = points[depot_idx]
        customers = clusters.get(depot_idx, [])
        vehicles = vehicles_per_depot[idx] if vehicles_per_depot else args.vehicles
        routes, km, minutes = solve_depot_vrp_ga(
            depot_idx=depot_idx,
            customers=customers,
            vehicles=vehicles,
            dist=dist,
            duration_matrix=duration_matrix,
            rng=rng,
            pop_size=args.pop_size,
            generations=args.generations,
            crossover_p=args.crossover_p,
            mutation_p=args.mutation_p,
            cut_mutation_p=args.cut_mutation_p,
            load_balance_weight=max(0.0, args.load_balance_weight),
            objective=args.objective,
            duration_weight=max(0.0, args.duration_weight),
        )

        results['total_km'] += km
        results['total_minutes'] += minutes
        route_entries = []
        for ri, route in enumerate(routes, start=1):
            forward_km = route_distance_km(depot_idx, route, dist)
            forward_minutes = route_duration_minutes(depot_idx, route, duration_matrix)
            route_entries.append(
                {
                    'route_idx': ri,
                    'stops': [points[c].id for c in route],
                    'stops_idx': route,
                    'customers': len(route),
                    'km': forward_km,
                    'minutes': forward_minutes,
                }
            )

        results['depots'].append(
            {
                'depot': {'idx': depot_idx, 'id': depot_point.id, 'address': depot_point.address},
                'vehicles': min(max(1, vehicles), max(1, len(customers))),
                'customers': len(customers),
                'km': km,
                'minutes': minutes,
                'routes': route_entries,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Toplam km: {results['total_km']:.3f}")
    if results['total_minutes']:
        print(f"Toplam sure: {results['total_minutes']:.1f} dk (~{results['total_minutes']/60:.2f} saat)")
    for depot_info in results['depots']:
        label = depot_info['depot']['id']
        print(f"\nDepo {label} icin rotalar:")
        for route in depot_info['routes']:
            km = float(route.get('km', 0.0))
            minutes = float(route.get('minutes', 0.0) or 0.0)
            cust = route.get('customers', len(route.get('stops_idx', [])))
            print(f"  Rota {route['route_idx']}: {km:.1f} km, {minutes:.1f} dk, {cust} musteri")
    print(f'Cikti: {out_path}')

    if args.plot_html:
        save_routes_map(points, results, Path(args.plot_html))
    if args.plot_km_chart:
        save_depot_km_chart(results, Path(args.plot_km_chart))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
