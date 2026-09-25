# EDA — train-only noise statistics

## Names (train S1)
- rows: 2,206,821
- `&` in name: 111,897 (5.07%) | word `and`: 57,296
- Devanagari script names: 0 (0.00% of all; 0.00% of India)
- domain/DBA-style names (www/.com/.in): 0 (0.00%)
- very short names (<4 chars): 561 | digit/punct-only: 0

### Top name-final tokens (legal-suffix candidates)
`limited`:521,915 `llc`:355,736 `inc`:238,306 `ltd`:148,595 `c`:45,557 `llp`:39,725 `corp`:33,761 `group`:32,678 `pc`:25,827 `co`:25,191 `center`:22,269 `associates`:22,165 `pllc`:20,531 `partners`:19,351 `clinic`:18,107 `care`:16,845 `lp`:16,477 `corporation`:13,540 `company`:12,316 `trust`:9,924 `holdings`:8,309 `society`:7,024 `school`:5,913 `foundation`:5,313 `church`:5,114

### Top name-final bigrams
`private_limited`:431,869 `pvt_ltd`:121,461 `p_c`:22,040 `l_c`:21,843 `&_co`:10,105 `associates_llc`:7,745 `partners_llc`:6,325 `associates_inc`:5,230 `center_llc`:4,780 `public_limited`:4,496 `partners_inc`:3,996 `care_llc`:3,760 `dds_pc`:3,742 `india_limited`:3,521 `clinic_llc`:3,279 `specialists_llc`:3,243 `medicine_llc`:3,241 `center_inc`:3,207 `care_associates`:3,120 `health_llc`:3,059

## Addresses (train S2+S3)
- rows: 10,320,219
- US rows w/ 5-digit ZIP-like: 670,930 (10.84% of US rows)
- India rows w/ 6-digit PIN-like: 6,060 (0.15% of India rows)
- landmark refs (near/opp/behind/nr): 416,071 (4.03%)
- municipal numbering (1-11-251/1B style): 1,148,226 (11.13%)

### Top address tokens
`no`:2,191,549 `road`:1,307,985 `delhi`:938,424 `new`:699,557 `street`:696,186 `st`:667,808 `rd`:658,744 `city`:658,629 `1`:632,396 `floor`:599,795 `nagar`:589,896 `dr`:565,438 `a`:560,643 `c`:531,858 `2`:489,155 `mumbai`:459,873 `drive`:457,801 `ave`:448,317 `north`:407,395 `avenue`:406,005 `west`:401,195 `b`:389,014 `h`:377,523 `plot`:347,329 `maharashtra`:344,331 `o`:326,795 `mh`:323,065 `tx`:307,612 `bangalore`:302,190 `texas`:295,943

## Ground-truth structure
- matched S2/S3 ids referenced by >1 S1 entity: 0 of 7,638,365 (max S1 per match: 1)

## Cross-source name noise (measured)
| file | rows | Devanagari | domain-style |
|---|---|---|---|
| train_source1.tsv | 2,206,821 | 0 (0.00%) | 0 (0.00%) |
| train_source2.tsv | 5,034,616 | 269,424 (5.35%) | 201,266 (4.00%) |
| train_source3.tsv | 5,285,603 | 158,003 (2.99%) | 210,922 (3.99%) |

**Key implication:** S1 is the clean Latin-only reference; Devanagari and domain/DBA noise live in S2/S3. Normalization must transliterate Devanagari → Latin and strip www/TLDs so cross-source pairs become comparable.
**Unique assignment:** every S2/S3 record matches at most one S1 entity — the match graph is a forest. The decision engine may exploit candidate-competition.
