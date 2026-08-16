# Annotation Guidelines — Gold Set

**Purpose:** This document specifies how to annotate the gold evaluation set for the distributed teacher-student NER research. The gold set is the human-annotated ground truth used to evaluate both LLM teachers (Sonnet 4.6, Haiku 4.5) and the six distilled student models in the Critic stage (Phase 7).

**Scope:** 515 postings drawn from the shared dev split. Annotated once; used for both teachers' Critic evaluations.

**Principle:** Annotate blind. Do not view the teacher labels during the annotation pass. Compare afterwards.

---

## Entity Types (8)

Identical schema to the teacher prompt (`pipeline/agents/prompts/teacher_system.txt`). Keep consistency.

### 1. SKILL

Technical skills, tools, programming languages, frameworks, libraries, platforms, methodologies, or domain-specific competencies **required or desired** for the role.

**Include:**
- Programming languages: `Python`, `Java`, `Go`, `TypeScript`, `SQL`
- Frameworks: `React`, `TensorFlow`, `Spring Boot`, `Django`
- Platforms and tools: `Kubernetes`, `Docker`, `Figma`, `Jira`, `Salesforce`
- Methodologies: `Agile`, `Scrum`, `Kanban`, `TDD`, `CI/CD`
- Soft skills *only when explicitly listed as a requirement*: `communication skills`, `leadership`, `problem-solving`
- Clinical/medical skills: `patient care`, `triage`, `wound care`
- Business skills: `financial modelling`, `stakeholder management`, `P&L ownership`

**Exclude:**
- Generic activities like `work`, `help`, `support` unless framed as a named skill
- Vague soft skills not in a requirements context (e.g. `we are a passionate team` — passion is not a skill here)

**Multiple occurrences:** if `Python` appears in three places, extract each occurrence separately.

### 2. JOB_TITLE

Role names for the posting itself OR other roles mentioned as career paths, team structure, or prerequisites.

**Include:**
- The posted role: `Senior Backend Engineer`, `Data Scientist`, `Assistant Branch Manager`
- Reporting structure: `reports to the Engineering Manager` → `Engineering Manager`
- Career path mentions: `path to Principal Engineer` → `Principal Engineer`
- Full seniority + role: extract `Senior Product Manager` as one span, not just `Product Manager`

**Span boundary rule:** include the seniority adjective as part of the JOB_TITLE; tag the seniority separately as EXPERIENCE_LEVEL (overlapping is allowed).

### 3. COMPANY

The hiring organisation's name and any subsidiary, parent, or named team mentioned as the employer.

**Include:**
- `Reply`, `Amazon Web Services`, `McKinsey & Company`, `Prisma Health`
- Team or division names presented as the employer: `AWS Professional Services`, `Google DeepMind`
- Abbreviations used as the company's own shorthand: `(AWS)` after `Amazon Web Services`

**Exclude:**
- Third-party companies named as customers, partners, or vendors (`our clients include Acme Corp`)
- Product names (`Salesforce` is a SKILL here, not a COMPANY, unless the company IS Salesforce)

**Disambiguation from CERT:** `NMLS` is a licensing system (CERT). `CFB` (Citizens First Bank) is a COMPANY. Prefer the context of how the acronym is introduced in the posting.

### 4. LOCATION

Geographic locations, work modality markers, and timezone-based location constraints.

**Include:**
- Cities: `Casablanca`, `London`, `San Francisco`
- Regions and states: `California`, `EMEA`, `New England`
- Countries: `Morocco`, `United Kingdom`, `Germany`
- Modality markers: `Remote`, `Hybrid`, `Onsite`, `Work from home`, `WFH`
- Multi-location strings: extract each location as a separate entity (`London`, `Berlin`, `remote`)
- **Timezones when they function as a work-location constraint**: `EST`, `PST`, `CET`, `GMT+1`, `UTC-5`, `ET`, `PT`. Rationale: these carry the same semantic role as `Remote` or `EMEA` — they narrow where the employee can work from.

**Exclude:**
- Non-workplace addresses (customer sites, convention venues)
- Pure schedule metadata that is not a location constraint (e.g. `9-5 ET` in an office-based role describing the shift, where the office location is specified elsewhere)

### 5. EXPERIENCE_LEVEL

Seniority indicators, years-of-experience requirements, career-stage labels.

**Include:**
- Seniority labels: `Junior`, `Mid`, `Senior`, `Lead`, `Principal`, `Staff`, `Entry-level`, `Graduate`
- Years of experience phrases: `5+ years`, `at least 3 years of experience`, `minimum 7 years`
- Career stage: `early-career`, `mid-career`, `experienced professional`

**Overlapping entities:** when a title includes a seniority marker, tag both — `Senior Engineer` is JOB_TITLE (the full span) AND `Senior` is EXPERIENCE_LEVEL (the substring). Overlapping is expected.

### 6. EDUCATION

Degree requirements and educational qualifications.

**Include:**
- Degrees: `Bachelor's Degree`, `Master's`, `PhD`, `BSc`, `MSc`, `MBA`, `JD`, `Doctorate`, `Associate Degree`
- Degree + field: `BSc in Computer Science`, `Master's in Physical Therapy` — extract the full span
- Equivalencies: `Bachelor's or equivalent experience`

**Span boundary rule:** prefer the fullest sensible span. `Bachelor's Degree in Computer Science` is one EDUCATION entity (not two).

### 7. CERT

Professional certifications, licences, and credentials.

**Include:**
- Technology certifications: `AWS Solutions Architect`, `PMP`, `Scrum Master`, `CISSP`, `CKA`
- Medical/clinical certifications: `RN`, `BLS`, `ACLS`, `PALS`, `CPR`, `Board Certified`
- State licences: `Licensed Physical Therapist`, `CPA`, `Bar Admitted`
- Vendor-specific: `Oracle Certified Professional`, `Microsoft Certified: Azure Solutions Architect`

### 8. COMPENSATION

Salary ranges, equity, bonuses, and benefit mentions.

**Include:**
- Explicit salary: `$150,000 - $180,000`, `120k USD`, `competitive salary`
- Equity and bonus: `stock options`, `equity grant`, `sign-on bonus`, `annual bonus`
- Benefits: `healthcare`, `401(k)`, `medical, dental, vision`, `unlimited PTO`, `paid time off`, `parental leave`
- Perks framed as compensation: `gym membership`, `remote work stipend`

---

## General Rules

### R1. Text verbatim
Every entity `text` must be a **literal substring** of the source posting. Do not paraphrase, expand abbreviations, normalise casing, or correct typos.

### R2. Exact offsets
`start` is the inclusive character offset; `end` is the exclusive offset (Python-slice convention). `source_text[start:end]` must equal `text`.

### R3. Preserve duplicates
If the same phrase appears multiple times in different contexts, annotate each occurrence as a separate entity. Do not deduplicate.

### R4. Overlapping entities
Overlapping entities of **different types** are allowed and encouraged (e.g. `Senior Engineer` is both JOB_TITLE and contains an EXPERIENCE_LEVEL). Overlapping entities of the **same type** should be deduplicated — annotate only the longer span.

### R5. Preserve casing and punctuation
Match the source exactly. `iOS` stays lowercase-i, `JavaScript` stays one word, `C++` keeps the pluses.

### R6. Long-form AND short-form
Extract both. `Amazon Web Services (AWS)` gives two COMPANY entities: `Amazon Web Services` and `AWS`.

### R7. Not-an-entity decisions
Err on the side of recall during the main pass. The gold set is for evaluation; including borderline cases gives a truer picture of what's extractable than filtering aggressively.

### R8. Empty is valid
If a posting has no entities of a given type, skip that type. Postings with zero entities are rare but legal — save them with an empty entity list.

---

## Calibration Pilot (Step 3 of the workflow)

Before starting the main 515-posting annotation:

1. Annotate 20 postings blind.
2. Open each teacher's `dev.parquet` and compare your annotations to Sonnet and Haiku on the same 20 postings.
3. Identify patterns where you systematically differ:
   - Do you split `Bachelor's Degree in CS` where Sonnet merges?
   - Do you skip soft skills that Haiku extracts?
   - Do you tag `communication` as SKILL where Sonnet doesn't?
4. Update these guidelines with the resolved rules.
5. Discard the pilot annotations (don't carry them into the main set — you'll re-annotate those 20 consistently with the updated guidelines).

---

## Intra-Annotator Stability (Step 5 of the workflow)

After the main 515-posting pass, wait ≥ 1 week. Then:

1. Re-annotate 50 random postings (use a fixed seed for reproducibility).
2. Compute F1 of your re-annotations vs your original annotations.
3. Report as *intra-annotator F1* in the paper. A value above 0.85 is considered reliable.

Low intra-annotator F1 suggests the guidelines need tightening or the annotator was inconsistent (fatigue, learning effects).

---

## Edge Cases and Disambiguation

Discovered during the pilot (2026-04-14) and the scale runs. Add new rules here as they arise during annotation.

### Acronym disambiguation

LLMs (especially Haiku) occasionally misclassified acronyms. Examples from the runs:

- `CFB` in a Citizens First Bank posting → COMPANY (the bank's shorthand), not CERT
- `NMLS` → CERT (licensing system)
- `AWS` → SKILL in most contexts; COMPANY only if the employer is Amazon/AWS

**Rule:** determine classification by context — is the acronym introduced as part of the hiring company's name, or as a requirement the candidate should have?

### Degree + field

Prefer the fullest sensible span.

- `Associate's Degree in Business` → one EDUCATION entity (do not split)
- `Bachelor's or equivalent` → one EDUCATION entity (include "or equivalent")

### Compound job titles

- `Senior Backend Engineer` → one JOB_TITLE entity (`Senior Backend Engineer`) AND `Senior` tagged as EXPERIENCE_LEVEL (overlapping)
- `Staff Software Engineer, Machine Learning` → one JOB_TITLE entity (the full span)

### Location lists

Multiple locations in a single string (`London, Berlin, or Remote`) → three LOCATION entities.

### Timezones as location constraints

Added 2026-04-15. Timezone identifiers in job postings almost always function as work-location constraints. Tag as LOCATION.

| Example | Action |
|---------|--------|
| `"Remote, EST only"` | `EST` → LOCATION, `Remote` → LOCATION (two entities) |
| `"work CET hours"` | `CET` → LOCATION |
| `"covers GMT+1 to GMT+3"` | Three LOCATION entities |
| `"must be available during EST business hours for customer calls"` | `EST` → LOCATION (marginal; lean inclusive) |
| `"9-5 ET shift at our Boston office"` | `ET` → do not tag (pure schedule metadata); `Boston` → LOCATION |

**Why this matters for downstream metrics:** the two LLM teachers (Sonnet 4.6 and Haiku 4.5) labelled without explicit timezone guidance. Expect inconsistency between teachers on this edge case. This reduces teacher-vs-gold F1 but is a valid finding about LLM teacher schema drift on edge cases — flag it in the paper rather than try to reconcile.

### Soft skills threshold

Soft skills in a **requirements list** (`must have excellent communication skills`) → SKILL.
Soft skills as **marketing copy** (`join our passionate team`) → not a SKILL.

---

## Format for gold.parquet

After annotation, `pipeline/training/{teacher}/gold.parquet` has columns:

| Column | Type | Description |
|--------|------|-------------|
| `job_link` | string | Join key |
| `job_summary` | string | Source text (read-only) |
| `entities` | list<struct> | Annotator's entity spans |

Each entity struct has `text`, `type`, `start`, `end` — identical to the teacher output schema.

---

## Provenance

- Author: Achraf SOLTANI
- Date: April 2026
- Guidelines version: 1.0
- Annotation pass: blind (no teacher labels shown)
- Tool: Label Studio (self-hosted via Podman)

Report any issues discovered during annotation as a new section below; do not silently retrofit rules.
