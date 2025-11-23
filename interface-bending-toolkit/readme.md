# NBM Metadata Toolkit — README

This repository provides a small, extensible toolkit for managing and searching the **Neural Bending Manual (NBM)** metadata index.  
It includes:

- `nbm_master_metadata_with_stages.yml` — master annotated bibliography + metadata  
- `nbm_search.py` — CLI search tool with LaTeX/PDF export  
- `nbm_annotated_bibliography_v2.pdf` — typeset annotated bibliography  
- `README.md` — documentation

The tools enable cross-referencing bibliography sources across the NBM **axes**, **modes**, **stages**, **tags**, and **keywords**, forming a structured, searchable research index for the dissertation.

---

## Repository Structure

.
├── nbm_master_metadata_with_stages.yml
├── nbm_search.py
├── nbm_annotated_bibliography_v2.pdf
└── README.md

markdown
Copy code

---

# 1. Master Metadata File

### `nbm_master_metadata_with_stages.yml`

The master YAML file serves as the core database for the Neural Bending Manual.  
Each entry includes:

- **author**
- **title**
- **year**
- **summary**
- **relation_to_nbm**
- **use_in_nbm**
- **axis**
- **mode**
- **stages**
- **tags**
- **keywords**

These attributes allow multidimensional indexing across technical, philosophical, and tactical dimensions of neural bending practice.

---

# 2. Search CLI: `nbm_search.py`

A command-line utility for querying the metadata and generating output.

### Basic usage

```bash
python nbm_search.py --query "cybernetics"
Filters
bash
Copy code
--query / -q    # full-text search across most fields
--axis / -a     # interface, latent, substrate, governance
--mode / -m     # tactical, performative, creative, critical, etc.
--tag / -t      # tag-based filtering
--stage / -s    # Stage number or label (e.g. 5, "Stage 5")
Examples
Search by conceptual dimension:

bash
Copy code
python nbm_search.py --axis latent --mode tactical
Search all items tied to Stage 7 (Memory Excavation):

bash
Copy code
python nbm_search.py --stage 7
Keyword search:

bash
Copy code
python nbm_search.py --query recursion
3. Exporting LaTeX Bibliographies
The CLI can output search results as a standalone LaTeX document:

bash
Copy code
python nbm_search.py --query "recursion" --latex recursion_bib.tex
This produces a full article-style LaTeX file containing:

Section headings per entry

Summary

Relation to NBM

Use in Manual

Axis / Mode / Stage metadata

Auto-compile to PDF
If LaTeX (pdflatex) is installed:

bash
Copy code
python nbm_search.py --axis governance --latex gov_bib.tex --latex-pdf
If LaTeX is unavailable, the .tex file will still be generated.

4. Annotated Bibliography PDF
The included file:

Copy code
nbm_annotated_bibliography_v2.pdf
is a typeset, page-safe annotated bibliography generated with ReportLab.

Features:

Proper word-wrapping

Safe page breaks (no text cutoff)

Clean serif typography

Heading + meta-line (Axis · Mode · Stage) per entry

Suitable for dissertation appendices or printing.

5. Dependencies
Install required Python libraries:

bash
Copy code
pip install reportlab pyyaml
Optional:

LaTeX installation (pdflatex) for PDF generation from the CLI’s LaTeX mode.

6. Extending the System
To add new entries, edit the YAML file directly.
Use the existing pattern:

yaml
Copy code
KeyID:
  author: "Author Name"
  title: "Title of Work"
  year: 2020
  summary: "Brief summary..."
  relation_to_nbm: "How this source informs the manual..."
  use_in_nbm: "Where it appears in stages, patches, or appendices..."
  axis: ["latent"]
  mode: ["tactical"]
  stages: ["Stage 5 — Representation Disruption"]
  tags: ["media-archaeology", "latent"]
  keywords: ["manifold", "curvature", "drift"]
The CLI will automatically incorporate the updated metadata.

7. License
MIT License (recommended).
Feel free to replace with your preferred research or open-source license.

8. Attribution
Toolkit authored collaboratively within the Limits of Ctrl research environment.
If cited:

Neural Bending Metadata Toolkit (2025). Internal research infrastructure for the Neural Bending Manual and Limits of Ctrl project.

less
Copy code

If you want:

- a GitHub-friendly version with shields/badges  
- a dissertation-appendix version  
- a minimalist or academic-tone version  

I can generate those instantly.