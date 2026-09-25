# Data profile (S1 validation)

## train_source1
- rows: 2206821 (unique ids: 2206821)
- countries: `{'US': 1323633, 'India': 883188}`
- empty name/address: 0 / 0
- name len p50/p90/p99/max: {'p50': 24, 'p90': 34, 'p99': 42, 'max': 105, 'mean': 24.03}
- addr len p50/p90/p99/max: {'p50': 41, 'p90': 90, 'p99': 124, 'max': 256, 'mean': 52.07}
- errors: 0

## train_source2
- rows: 5034616 (unique ids: 5034616)
- countries: `{'US': 3016817, 'India': 2017799}`
- empty name/address: 0 / 168967
- name len p50/p90/p99/max: {'p50': 25, 'p90': 37, 'p99': 48, 'max': 104, 'mean': 25.1}
- addr len p50/p90/p99/max: {'p50': 37, 'p90': 83, 'p99': 118, 'max': 249, 'mean': 46.23}
- errors: 0

## train_source3
- rows: 5285603 (unique ids: 5285603)
- countries: `{'US': 3170056, 'India': 2115547}`
- empty name/address: 0 / 175916
- name len p50/p90/p99/max: {'p50': 25, 'p90': 37, 'p99': 50, 'max': 123, 'mean': 25.2}
- addr len p50/p90/p99/max: {'p50': 42, 'p90': 77, 'p99': 115, 'max': 240, 'mean': 46.71}
- errors: 0

## test_source1
- rows: 1732544 (unique ids: 1732544)
- countries: `{'India': 809986, 'US': 663106, 'France': 259452}`
- empty name/address: 0 / 0
- name len p50/p90/p99/max: {'p50': 24, 'p90': 34, 'p99': 42, 'max': 92, 'mean': 23.84}
- addr len p50/p90/p99/max: {'p50': 50, 'p90': 93, 'p99': 126, 'max': 268, 'mean': 57.21}
- errors: 0

## test_source2
- rows: 4887273 (unique ids: 4887273)
- countries: `{'India': 2312565, 'US': 1871330, 'France': 703378}`
- empty name/address: 0 / 129408
- name len p50/p90/p99/max: {'p50': 25, 'p90': 38, 'p99': 49, 'max': 102, 'mean': 25.7}
- addr len p50/p90/p99/max: {'p50': 43, 'p90': 87, 'p99': 120, 'max': 269, 'mean': 50.41}
- errors: 0

## test_source3
- rows: 5082316 (unique ids: 5082316)
- countries: `{'India': 2405000, 'US': 1945701, 'France': 731615}`
- empty name/address: 0 / 136098
- name len p50/p90/p99/max: {'p50': 25, 'p90': 38, 'p99': 50, 'max': 103, 'mean': 25.66}
- addr len p50/p90/p99/max: {'p50': 43, 'p90': 81, 'p99': 117, 'max': 267, 'mean': 48.74}
- errors: 0

## train_ground_truth
- rows: 2206821
- total matches: 7638365
- empty (singleton) lists: 123247
- cardinality buckets (n_matches -> count, 11 => >10): `{0: 123247, 1: 119157, 2: 375212, 3: 530841, 4: 484115, 5: 321957, 6: 164868, 7: 63968, 8: 18680, 9: 4205, 10: 534, 11: 37}`
- errors: 0

**GATE: PASS** (0 total errors)
