# TrialMine vs ClinicalTrials.gov — Head-to-Head

Comparison of TrialMine's full pipeline (BM25 + Semantic + RRF + Cross-Encoder + LightGBM) against ClinicalTrials.gov's native v2 keyword search API across 20 diverse oncology queries spanning seven categories. Top-20 from each side.

## Headline

- **Average overlap per query**: 0.5 / 20
- **Average ours-only per query**: 19.6
- **Of those, judged ≥2 by Haiku**: 15.1 (77% precision on the disjoint set)
- **Average theirs-only per query**: 13.5
- **Of CT.gov-only, judged ≥2 by Haiku**: 9.1 (67% precision on the disjoint set)
- **Precision delta (ours − theirs)**: **+9.8%** on the disjoint set

## Per-query summary

| Category | Query | ∩ | Ours-only | Ours-rel ≥2 | Theirs-only | Theirs-rel ≥2 |
|---|---|---:|---:|---:|---:|---:|
| common | breast cancer HR+ CDK4/6 inhibitor | 2 | 18 | 18 | 18 | 16 |
| common | NSCLC brain mets | 0 | 20 | 17 | 9 | 9 |
| common | ovarian BRCA mutation | 1 | 19 | 18 | 19 | 15 |
| rare | angiosarcoma scalp | 0 | 20 | 12 | 0 | 0 |
| rare | merkel cell carcinoma | 0 | 20 | 20 | 20 | 17 |
| rare | GIST imatinib resistant | 1 | 19 | 15 | 19 | 15 |
| pediatric | medulloblastoma 6 year old | 1 | 19 | 13 | 8 | 4 |
| pediatric | wilms tumor relapsed | 1 | 19 | 17 | 19 | 3 |
| complex | 58M EGFR exon 19 NSCLC failed osimertinib phase 2-3 | 0 | 20 | 5 | 0 | 0 |
| complex | 45F TNBC completed AC-T 3mo ago | 0 | 20 | 0 | 0 | 0 |
| complex | 62F HER2+ metastatic breast post-trastuzumab progression | 0 | 20 | 14 | 0 | 0 |
| vague | I have cancer what trials | 0 | 20 | 15 | 20 | 14 |
| vague | mom has bone cancer | 0 | 20 | 14 | 1 | 0 |
| vague | just diagnosed need help | 0 | 20 | 12 | 20 | 5 |
| geographic | pancreatic cancer trials Texas | 0 | 20 | 17 | 20 | 15 |
| geographic | trials at MD Anderson | 0 | 20 | 18 | 20 | 11 |
| treatment | CAR-T lymphoma | 0 | 20 | 20 | 20 | 14 |
| treatment | PARP inhibitor ovarian | 3 | 17 | 17 | 17 | 11 |
| existing | triple negative breast cancer neoadjuvant | 0 | 20 | 20 | 20 | 18 |
| existing | targeted therapy for EGFR mutated lung cancer | 0 | 20 | 20 | 20 | 15 |

## By category (averages per query)

| Category | n | overlap | ours-only | ours-rel | theirs-only | theirs-rel |
|---|---:|---:|---:|---:|---:|---:|
| common | 3 | 1.0 | 19.0 | 17.7 | 15.3 | 13.3 |
| complex | 3 | 0.0 | 20.0 | 6.3 | 0.0 | 0.0 |
| existing | 2 | 0.0 | 20.0 | 20.0 | 20.0 | 16.5 |
| geographic | 2 | 0.0 | 20.0 | 17.5 | 20.0 | 13.0 |
| pediatric | 2 | 1.0 | 19.0 | 15.0 | 13.5 | 3.5 |
| rare | 3 | 0.3 | 19.7 | 15.7 | 13.0 | 10.7 |
| treatment | 2 | 1.5 | 18.5 | 18.5 | 18.5 | 12.5 |
| vague | 3 | 0.0 | 20.0 | 13.7 | 13.7 | 6.3 |

## Strongest ours-only wins per query

Trials TrialMine surfaced that CT.gov's keyword search did NOT, ranked by Haiku-judged relevance.

### common — `breast cancer HR+ CDK4/6 inhibitor`

- **NCT06207734** (rel=3) — Discontinuation of CDK4/6 Inhibitors in Patients With Metastatic HR Positive, HER2 Negative Breast Cancer With Durable Disease Control: A Ra
  > Exact match for HR+ breast cancer with CDK4/6 inhibitor treatment; patient meets inclusion criteria for ER+/HER2- metastatic disease.
- **NCT07227233** (rel=3) — Artificial Intelligence and Machine Learning-Enhanced Biomarker-dRiven CDK4/6 Inhibitor Rechallenge in HR+ HER2- Advanced Breast Tumors.
  > Excellent match: directly addresses HR+ HER2- advanced breast cancer treatment with CDK4/6 inhibitors, and patient meets basic eligibility criteria (age, cancer type, receptor status).
- **NCT07213206** (rel=3) — Chemotherapy Omission in HR-positive/HER2-positive Breast Cancer With Lymph Node Negative Disease Receiving Adjuvant Endocrine Therapy and C
  > Strong match: HR-positive breast cancer with CDK4/6 inhibitor is the primary focus, though this trial also requires HER2-positivity and early-stage disease (N0), which may or may not apply to the patient.

### common — `NSCLC brain mets`

- **NCT02385136** (rel=3) — Temozolomide and Concomitant Whole Brain Radiotherapy in NSCLC Patients With Brain Metastases: A Randomized Trial
  > Exact match for NSCLC with brain metastases, directly addresses the patient's search query with specific treatment intervention and relevant eligibility criteria.
- **NCT05465343** (rel=3) — Efficacy and Safety of Furmonertinib in Patients With EGFR Mutations in Advanced NSCLC With Brain Metastases: A Prospective, Open-label, Pha
  > Exact match for patient's NSCLC with brain metastases condition; trial specifically enrolls EGFR-mutant NSCLC patients with brain mets and evaluates a targeted TKI therapy.
- **NCT02132598** (rel=3) — A Single-Arm Phase II Clinical Trial of Cabozantinib (XL184) in Patients With Previously Treated Non-Small Cell Lung Cancer (NSCLC) With Bra
  > Exact match for patient's condition (NSCLC with brain metastases) and trial directly addresses this indication with appropriate inclusion criteria for previously treated patients.

### common — `ovarian BRCA mutation`

- **NCT03063710** (rel=3) — Olaparib Expanded Access Program for BRCA Mutated Platinum Sensitive Relapsed High Grade Epithelial Ovarian Cancer Patients
  > Excellent match: trial specifically targets ovarian cancer patients with BRCA mutations, directly addressing the patient's search query, though trial is no longer available for enrollment.
- **NCT01844986** (rel=3) — A Phase III, Randomised, Double Blind, Placebo Controlled, Multicentre Study of Olaparib Maintenance Monotherapy in Patients With BRCA Mutat
  > Perfect match: directly targets ovarian cancer patients with BRCA mutations receiving maintenance therapy after first-line platinum chemotherapy, aligning precisely with the patient's search query.
- **NCT02225015** (rel=3) — Cancer Prevention in Women With a BRCA Mutation: A Follow-up Genetic Counselling Intervention
  > Directly addresses ovarian cancer prevention in BRCA mutation carriers with inclusive eligibility criteria that would accommodate a patient with ovarian BRCA mutations.

### rare — `angiosarcoma scalp`

- **NCT04518124** (rel=3) — Neoadjuvant Trial on the Efficacy of Propranolol Monotherapy in Cutaneous Angiosarcoma
  > Perfect match: patient has angiosarcoma of the scalp (cutaneous location), trial specifically targets cutaneous angiosarcoma with no anatomical restrictions, and patient would likely meet all stated eligibility criteria.
- **NCT05799612** (rel=3) — Phase I Study of TH1 Dendritic Cell Immunotherapy for the Treatment of Cutaneous Angiosarcoma
  > Exact match for cutaneous angiosarcoma with specific inclusion of head & neck location matching scalp, and patient would likely meet eligibility criteria for this immunotherapy trial.
- **NCT03544567** (rel=3) — A Phase 2 Study of Oraxol in Subjects With Cutaneous Angiosarcoma
  > Angiosarcoma of the skin is directly relevant to scalp angiosarcoma, and the trial's inclusion criteria for histologically-confirmed cutaneous angiosarcoma not amenable to surgery aligns well with typical scalp presentations.

### rare — `merkel cell carcinoma`

- **NCT04975152** (rel=3) — Neoadjuvant Cemiplimab in Newly Diagnosed or Recurrent Stage I-II Merkel Cell Carcinoma and Locoregionally Advanced Cutaneous Squamous Cell 
  > Direct match - trial specifically targets Merkel cell carcinoma, the exact condition in the patient's search query, with clearly defined eligibility criteria for newly diagnosed or recurrent Stage I-II disease.
- **NCT04792073** (rel=3) — A Phase II Single-Arm Clinical Trial Assessing Comprehensive Ablative Radiation Therapy With Avelumab in Unresectable and Metastatic Merkel 
  > Perfect match - trial directly addresses merkel cell carcinoma with specific inclusion criteria for unresectable/metastatic disease that patient is searching for.
- **NCT05594290** (rel=3) — Window-of-opportunity Study of Chemo-immunotherapy in Patients With Resectable Merkel Cell Carcinoma Prior to Surgery: the MERCURY Trial
  > Trial directly addresses merkel cell carcinoma with chemo-immunotherapy in resectable disease, matching the patient's search query exactly with well-defined inclusion criteria.

### rare — `GIST imatinib resistant`

- **NCT01735968** (rel=3) — A Dose-finding Phase Ib Multicenter Study of Imatinib in Combination With the Oral Phosphatidyl-inositol 3-kinase (PI3K) Inhibitor BYL719 in
  > Excellent match: patient has imatinib-resistant GIST and trial specifically enrolls GIST patients who failed prior imatinib therapy, testing imatinib in combination with a novel PI3K inhibitor.
- **NCT04138381** (rel=3) — A Multicenter, Phase Ib/II Trial of Selinexor as a Single Agent and in Combination With Imatinib in Patients With Metastatic and/or Unresect
  > Trial directly addresses imatinib-resistant GIST with explicit inclusion criteria requiring prior imatinib failure, making it a perfect match for the patient's search query.
- **NCT01089595** (rel=3) — Open Label Phase II Randomized Trial of Tasigna (Nilotinib) 400 mg Twice Daily Alone or in Combination With Gleevec (Imatinib Mesylate) 400 
  > Directly targets imatinib-resistant GIST patients with documented progression on high-dose imatinib, matching the patient's exact search criteria and eligibility requirements.

### pediatric — `medulloblastoma 6 year old`

- **NCT06942039** (rel=3) — A Pilot Study of Intrathecal Topotecan and Maintenance Chemotherapy in the Post-consolidation Setting for the Treatment of High-risk Embryon
  > Exact match: patient is 6 years old with medulloblastoma and trial specifically treats medulloblastoma in children ≤6 years old with intrathecal chemotherapy in post-consolidation setting.
- **NCT02724579** (rel=3) — A Phase 2 Study of Reduced Therapy for Newly Diagnosed Average-Risk WNT-Driven Medulloblastoma Patients
  > Perfect match: patient has medulloblastoma, is within the 3-22 year age range (6 years old), and trial specifically targets newly diagnosed WNT-driven medulloblastoma which is a common subtype in this age group.
- **NCT06959979** (rel=3) — Novel Molecular Targets and Innovative Therapeutic Perspective in Medulloblastoma
  > Direct match on medulloblastoma diagnosis in pediatric patient age group with inclusive eligibility criteria for all medulloblastoma patients treated at their facility.

### pediatric — `wilms tumor relapsed`

- **NCT05384821** (rel=3) — Phase 1-2 Trial Evaluating Metronomic Chemotherapy in Patients With a Relapsed or Refractory Wilms Tumor
  > Direct match: trial specifically targets relapsed/refractory Wilms tumor, which exactly matches the patient's search query for Wilms tumor relapse.
- **NCT02452554** (rel=3) — A Phase 2 Study of IMGN901 (Lorvotuzumab Mertansine; NSC#: 783609) in Children With Relapsed or Refractory Wilms Tumor, Rhabdomyosarcoma, Ne
  > Trial directly targets relapsed Wilms tumor with a Phase 2 immunotherapy agent and explicitly lists Wilms tumor as a primary stratum in inclusion criteria.
- **NCT00002610** (rel=3) — National Wilms Tumor Study-5 -- Treatment of Relapsed Patients, A National Wilms Tumor Study Group Phase III Study
  > Perfect match: patient is searching for relapsed Wilms tumor treatment and this trial specifically enrolls relapsed Wilms tumor patients with multiple histologic subtypes.

### complex — `58M EGFR exon 19 NSCLC failed osimertinib phase 2-3`

- **NCT04479306** (rel=3) — A Ph1b Study of Osimertinib + Alisertib or Sapanisertib for Osimertinib-Resistant EGFR Mutant Non-Small Cell Lung Cancer (NSCLC) (Crossover 
  > Excellent match: patient has EGFR exon 19 NSCLC with osimertinib failure, and trial specifically enrolls osimertinib-resistant EGFR mutant NSCLC patients with exon 19 deletions seeking alternative combination therapies.
- **NCT03755102** (rel=3) — A Pilot Study of Dacomitinib With or Without Osimertinib for Patients With Metastatic EGFR Mutant Lung Cancers With Disease Progressionon Os
  > Perfect match: 58M with EGFR exon 19 NSCLC who failed osimertinib meets all key inclusion criteria (EGFR mutation, metastatic NSCLC, prior osimertinib with progression, no prior first/second gen inhibitors).
- **NCT06363734** (rel=3) — Osimertinib Plus Dalpiciclib in Patients With EGFR-mutant, CDK4/6 Pathway Aberrant, Advanced Non-small Cell Lung Cancer Following Acquired R
  > Exact match: EGFR-mutant NSCLC patient with acquired resistance to third-generation EGFR TKI (osimertinib) seeking combination therapy, and trial explicitly recruits this population with osimertinib plus CDK4/6 inhibitor.

### complex — `62F HER2+ metastatic breast post-trastuzumab progression`

- **NCT00567879** (rel=3) — A Phase Ib/IIa Trial of Panobinostat in Combination With Trastuzumab in Adult Female Patients With HER2 Positive Metastatic Breast Cancer Wh
  > Perfect match: 62F HER2+ metastatic breast cancer patient with trastuzumab progression is exactly the target population for this trial testing panobinostat + trastuzumab combination therapy.
- **NCT03368729** (rel=3) — A Phase 1b/2 Study of the PARP Inhibitor Niraparib in Combination With Trastuzumab in Patients With Metastatic HER2+ Breast Cancer
  > Excellent match: patient has HER2+ metastatic breast cancer with trastuzumab progression, and trial specifically evaluates niraparib combined with trastuzumab in this exact population, offering a rational next-line therapeutic strategy.
- **NCT04307329** (rel=3) — Monalizumab and Trastuzumab In Metastatic HER2-pOSitive breAst Cancer: MIMOSA-trial
  > Highly relevant: patient has HER2+ metastatic breast cancer with trastuzumab progression, matches trial's focus on HER2+ metastatic disease post-HER2-directed therapy, and meets eligibility criteria for prior treatment lines.

### vague — `I have cancer what trials`

- **NCT01168206** (rel=3) — Assessment of Quality of Life and the Toxicity of Chemotherapy in Patients With Malignancies in Clinical Stages III and IV Under Palliative 
  > Trial directly matches patient's cancer diagnosis with advanced stage (III/IV) eligibility, chemotherapy/hormone therapy treatment options, and broad patient population requirements.
- **NCT04604158** (rel=3) — IIT2020-13-GRESHAM-ELLY: Evaluating the Effect of a Mobile Audio Companion (Elly) to Reduce Anxiety in Cancer Patients
  > Highly relevant: trial accepts any cancer type with current or recent cancer-targeted treatment, directly matching the patient's query about cancer trials.
- **NCT02096289** (rel=2) — A Phase I Trial Evaluating Oral Thioridazine in Combination With Intermediate Dose Cytarabine in Patients 55 Years and Older With Acute Myel
  > Patient has unspecified cancer but trial targets acute myeloid leukemia specifically; relevant if patient has AML, but unclear from vague query.

### vague — `mom has bone cancer`

- **NCT06008483** (rel=3) — A Dose Finding Study of CycloSam® (153-Sm-DOTMP) to Treat Solid Tumor(s) in the Bone or Metastatic to the Bone (Metastatic Prostate, Breast,
  > Trial directly addresses bone cancer and metastatic tumors to bone with broad eligibility criteria that would likely match a patient with bone cancer.
- **NCT04310410** (rel=3) — Feasibility of Combined Focused Ultrasound and Radiotherapy Treatment in Patients With Painful Bone Metastasis - the PRE-FURTHER Study -
  > The trial directly addresses bone cancer/bone metastases with a pain management focus, matching the patient's query, and appears to accept patients with bone cancer from any solid tumor type.
- **NCT06070259** (rel=3) — Examining Patient Involvement Patterns and Trends in Participation in Bone Cancer Clinical Trials
  > Direct match on bone cancer diagnosis with inclusive eligibility criteria and no prior treatment requirement that aligns with typical newly diagnosed patient scenarios.

### vague — `just diagnosed need help`

- **NCT06189261** (rel=3) — Feasibility and Acceptability of a Holistic Needs Assessment Intervention Employing Patient-reported Outcome Measures (PROMs) to Support New
  > Patient is newly diagnosed and seeking help; this trial specifically targets newly diagnosed melanoma patients (within 1 month post-diagnosis) with a holistic needs assessment intervention designed to support their needs.
- **NCT01468246** (rel=3) — Helping Ourselves, Helping Others: The Young Women's Breast Cancer Study
  > Strong match: patient with newly diagnosed breast cancer seeking help aligns perfectly with this observational study for young women with breast cancer that is actively enrolling.
- **NCT03317470** (rel=2) — Addressing Unmet Basic Needs to Improve Adherence Among Women With an Abnormal Pap
  > Patient with recent abnormal Pap diagnosis matches the trial's condition and newly-diagnosed status, but trial focuses on adherence intervention rather than treatment and is already completed.

### geographic — `pancreatic cancer trials Texas`

- **NCT01771146** (rel=3) — A Prospective Evaluation of Neoadjuvant FOLFIRINOX Regimen in Patients With Non-metastatic Pancreas Cancer (Baylor University Medical Center
  > Exact match for pancreatic cancer trials in Texas with clear inclusion criteria for adenocarcinoma patients and specific Texas locations (Baylor University Medical Center and Texas Oncology).
- **NCT01663272** (rel=2) — A Trial of Cabozantinib (XL184) and Gemcitabine in Advanced Pancreatic Cancer
  > Trial matches pancreatic cancer condition but location information is not provided in the trial details, so relevance to the Texas-specific search requirement cannot be confirmed.
- **NCT01587534** (rel=2) — Phase 2 Study of Randomized Controlled Trial of Pancreatic Enzyme Supplementation in Patients With Unresectable Pancreatic Cancer
  > Trial matches pancreatic cancer condition and patient eligibility criteria, but location information is not provided in the trial details to confirm Texas availability.

### geographic — `trials at MD Anderson`

- **NCT02106988** (rel=3) — Concurrent Chemotherapy and Radiation Therapy for Newly Diagnosed Patients With Stage I/II Nasal NK Cell Lymphoma
  > Trial is at MD Anderson (exact location match), recruiting for lymphoma patients, with clearly defined eligibility criteria that can be assessed against patient's medical profile.
- **NCT06259253** (rel=3) — Understanding Patient Experience Among Asians at MD Anderson
  > Trial is at MD Anderson (exact location match), accepts any cancer type, and patient demographics align with inclusion criteria if they are Asian/Asian American with prior inpatient cancer care at this institution.
- **NCT02515383** (rel=3) — Preliminary Testing of the MD Anderson Symptom Inventory (Adolescent Version)
  > Trial is located at MD Anderson Cancer Center as requested, enrolls adolescent cancer patients with hematologic and solid malignancies, and has active enrollment criteria matching typical cancer patients.

### treatment — `CAR-T lymphoma`

- **NCT07164560** (rel=3) — Clinical Study of TRBC1/2-Targeted CAR-T Cells in the Treatment of Relapsed/Refractory Peripheral T-Cell Lymphoma
  > Highly relevant: CAR-T cell therapy for lymphoma directly matches patient query, with specific focus on T-cell lymphoma subtype and active recruitment.
- **NCT03258047** (rel=3) — Novel Autologou CAR-T Therapy for Relapsed/Refractory B Cell Lymphoma
  > Direct match for CAR-T therapy in B cell lymphoma with relapsed/refractory disease, which is the standard indication for this treatment modality.
- **NCT04004637** (rel=3) — CD7 CAR-T Cells for Patients With Relapse/Refractory CD7+ NK/T Cell Lymphoma ,T-lymphoblastic Lymphoma and Acute Lymphocytic Leukemia
  > Direct match for CAR-T cell therapy in lymphoma with CD7+ NK/T cell lymphoma and T-lymphoblastic lymphoma explicitly listed as trial conditions.

### treatment — `PARP inhibitor ovarian`

- **NCT07472140** (rel=3) — To Develop and Implement The Scope of Medical Care for Homologous Recombination Deficient Ovarian Cancer, Fallopian-Tube Cancer, or Primary 
  > Direct match for PARP inhibitor treatment in ovarian cancer with homologous recombination deficiency, exactly aligned with patient's search query.
- **NCT02326844** (rel=3) — A Phase 2 Pilot Study of BMN 673 (Talazoparib), an Oral PARP Inhibitor, in Patients With Deleterious BRCA1/2 Mutation-Associated Ovarian Can
  > Perfect match: PARP inhibitor trial specifically for BRCA-associated ovarian cancer, directly addressing the patient's search query with relevant drug class and cancer type.
- **NCT04999605** (rel=3) — Phase Ib/II Clinical Study of Anti-PD-1 and VEGF Bispecific Antibody (AK112) Combined With PARP Inhibitor in the Treatment of Recurrent Ovar
  > Highly relevant: directly addresses PARP inhibitor treatment for ovarian cancer with matching cancer type and treatment approach, though trial is terminated.

### existing — `triple negative breast cancer neoadjuvant`

- **NCT04809779** (rel=3) — PD-1 Inhibitor Sintilimab Concurrent With Epirubicin Cyclophosphamide and Nab-paclitaxel as Neoadjuvant Therapy for Triple Negative Breast C
  > Perfect match: triple negative breast cancer with neoadjuvant chemotherapy is the exact condition and treatment approach the patient is searching for.
- **NCT04907344** (rel=3) — A Multicenter, Open, Randomized Controlled Study of Camrelizumab+ Nab-paclitaxel + Carboplatin Versus Nab-paclitaxel + Carboplatin as Neoadj
  > Exact match for triple negative breast cancer neoadjuvant therapy with Phase 2/3 trial specifically designed for this condition and treatment approach.
- **NCT03168880** (rel=3) — A Randomized Controlled Trial of Neoadjuvant Weekly Paclitaxel Versus Weekly Paclitaxel Plus Weekly Carboplatin In Women With Large Operable
  > Exact match: trial specifically studies neoadjuvant chemotherapy in triple negative breast cancer with clearly defined inclusion criteria for eligible patients.

### existing — `targeted therapy for EGFR mutated lung cancer`

- **NCT06018688** (rel=3) — A Phase II Study Evaluating Osimertinib Combined With Aspirin Neoadjuvant Therapy for Resectable EGFR Mutated Non-small Cell Lung Cancer (NS
  > Excellent match: directly addresses targeted therapy (osimertinib) for EGFR-mutated NSCLC with relevant phase II trial design and specific inclusion criteria for EGFR sensitizing mutations.
- **NCT06971406** (rel=3) — A Multicenter, Prospective Phase II Clinical Study of High-Dose Firmonertinib Combined With Bevacizumab and Intrathecal Pemetrexed in the Tr
  > Highly relevant match: directly addresses EGFR-mutated lung cancer with targeted therapy (firmonertinib), though specificity to leptomeningeal metastasis may limit applicability if patient lacks this complication.
- **NCT02886195** (rel=3) — EGFR-TKIs Combine Chemotherapy as First-line Therapy for Patients With Advanced EGFR Mutation-positive NSCLC
  > Trial directly addresses targeted therapy (EGFR-TKIs) combined with chemotherapy for advanced EGFR mutation-positive NSCLC, which is an exact match for the patient's search query regarding targeted therapy for EGFR mutated lung cancer.

## Caveats

- **CT.gov returns the entire registry (500K+ trials), not an oncology subset.** Some `theirs-only` results may be relevant trials we filtered out at ingest.
- **CT.gov's ranking is keyword-based, not ML-ranked**. The comparison is therefore TrialMine's ML pipeline vs a strong but unranked baseline, not vs a competing ML system.
- **Ours-only relevance is judged by Claude Haiku**, the same judge that built our training labels. This shares blind spots with the system under test. See `docs/evaluation-report.md` for the Sonnet-vs-Haiku kappa analysis.
