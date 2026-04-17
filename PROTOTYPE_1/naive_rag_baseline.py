"""
NAIVE RAG BASELINE 
SO3 objectives:
  1. Benchmark responses with ground truths
  2. BLEU, METEOR, ROUGE-1, ROUGE-L, BERTScore
  3. Hallucination detection
  4. Multi-model comparison
  5. Parameter tuning (temperature, max_tokens)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from bert_score import score as bert_score
import os
import re
import time
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from huggingface_hub import InferenceClient
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Download required NLTK data
nltk.download('wordnet',   quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4',   quiet=True)

# Token / Key check 
print("Checking API keys...")
print(f"  HF_TOKEN:        {'SET' if os.getenv('HF_TOKEN')         else 'NOT SET'}")
print(f"  GEMINI_API_KEY:  {'SET' if os.getenv('GEMINI_API_KEY')   else 'NOT SET'}")
print(f"  OPENAI_API_KEY:  {'SET' if os.getenv('OPENAI_API_KEY')   else 'NOT SET'}")
print()


# KNOWLEDGE BASE — DLSU CpE Checklist AY 2022-2023

DOCUMENTS = [
    # ── FIRST TERM ────────────────────────────────────────────────────────────
    {
        "id": "term1_overview",
        "text": "TERM 1 - First Term courses: NSTP101 (National Service Training Program-General Orientation, 0 units), FNDMATH (Foundation in Math FOUN, 5 units), BASCHEM (Basic Chemistry, 3 units), BASPHYS (Basic Physics, 3 units), FNDSTAT (Foundation in Statistics FOUN, 3 units), GEARTAP (Art Appreciation 2A, 3 units). Total: 17 units. No prerequisites required for First Term.",
        "metadata": {"term": "1", "type": "overview"}
    },
    # ── SECOND TERM ───────────────────────────────────────────────────────────
    {
        "id": "term2_overview",
        "text": "TERM 2 - Second Term courses: NSTPCW1 (National Service Training Program 1 2D, 3 units), GEMATMW (Mathematics in the Modern World 2A, 3 units), CALENG1 (Differential Calculus 1A, 3 units) requires FNDMATH as hard prerequisite, COEDISC (Computer Engineering as a Discipline 1E, 1 unit), PROLOGI (Programming Logic and Design Lecture 1E, 2 units), LBYCPA1 (Programming Logic and Design Laboratory 1E, 2 units) requires PROLOGI as co-requisite, LBYEC2A (Computer Fundamentals and Programming 1, 1 unit), GESTSOC (Science Technology and Society 2A, 3 units), GERIZAL (Life and Works of Rizal 2B, 3 units). Total: 18 units.",
        "metadata": {"term": "2", "type": "overview"}
    },
    {
        "id": "caleng1_prereq",
        "text": "CALENG1 (Differential Calculus) has FNDMATH as a hard prerequisite. Students must pass FNDMATH before enrolling in CALENG1.",
        "metadata": {"term": "2", "type": "prerequisite", "course": "CALENG1"}
    },
    {
        "id": "lbycpa1_coreq",
        "text": "LBYCPA1 (Programming Logic and Design Laboratory) requires PROLOGI as a co-requisite. Both PROLOGI and LBYCPA1 must be taken in the same term.",
        "metadata": {"term": "2", "type": "corequisite", "course": "LBYCPA1"}
    },
    # ── THIRD TERM ────────────────────────────────────────────────────────────
    {
        "id": "term3_overview",
        "text": "TERM 3 - Third Term courses: NSTPCW2 (National Service Training Program 2 2D, 3 units) requires NSTPCW1 as hard prerequisite, LCLSONE (Lasallian Studies 1, 1 unit), SAS1000 (Student Affairs Service 1000 LS, 0 units), LASARE1 (Lasallian Recollection 1, 0 units), ENGPHYS (Physics for Engineers 1B, 3 units) requires CALENG1 as soft/hard prerequisite and BASPHYS, LBYPH1A (Physics for Engineers Laboratory 1B, 1 unit) requires ENGPHYS as co-requisite, CALENG2 (Integral Calculus 1A, 3 units) requires CALENG1 as hard prerequisite, LBYCPEI (Object Oriented Programming Laboratory 1E, 2 units) requires PROLOGI as hard prerequisite, GEPCOMM (Purposive Communications 2A, 3 units), LCFAITH (Faith Worth Living, 3 units), GELECSP (Social Science and Philosophy 2B, 3 units). Total: 19 units.",
        "metadata": {"term": "3", "type": "overview"}
    },
    {
        "id": "engphys_prereq",
        "text": "ENGPHYS (Physics for Engineers) requires CALENG1 as a soft/hard prerequisite and BASPHYS. LBYPH1A (Physics for Engineers Laboratory) requires ENGPHYS as a co-requisite.",
        "metadata": {"term": "3", "type": "prerequisite", "course": "ENGPHYS"}
    },
    {
        "id": "caleng2_prereq",
        "text": "CALENG2 (Integral Calculus) requires CALENG1 as a hard prerequisite. Students must pass CALENG1 before enrolling in CALENG2.",
        "metadata": {"term": "3", "type": "prerequisite", "course": "CALENG2"}
    },
    # ── FOURTH TERM ───────────────────────────────────────────────────────────
    {
        "id": "term4_overview",
        "text": "TERM 4 - Fourth Term courses: CALENG3 (Differential Equations 1A, 3 units) requires CALENG2 as hard prerequisite, DATSRAL (Data Structures and Algorithms Lecture 1E, 1 unit) requires LBYCPEI as hard prerequisite, LBYCPA2 (Data Structures and Algorithms Laboratory 1E, 2 units) requires DATSRAL as co-requisite, DISCRMT (Discrete Mathematics 1E, 3 units) requires CALENG1 as hard prerequisite, FUNDCKT (Fundamentals of Electrical Circuits Lecture 1D, 3 units) requires ENGPHYS as hard prerequisite, LBYEC2M (Fundamentals of Electrical Circuits Lab 1D, 1 unit) requires FUNDCKT as co-requisite, ENGCHEM (Chemistry for Engineers 1B, 3 units) requires BASCHEM as hard prerequisite, LBYCH1A (Chemistry for Engineers Laboratory 1B, 1 unit) requires ENGCHEM as co-requisite, GEFTWEL (Physical Fitness and Wellness 2C, 2 units). Total: 19 units.",
        "metadata": {"term": "4", "type": "overview"}
    },
    {
        "id": "caleng3_prereq",
        "text": "CALENG3 (Differential Equations) requires CALENG2 as a hard prerequisite.",
        "metadata": {"term": "4", "type": "prerequisite", "course": "CALENG3"}
    },
    {
        "id": "datsral_prereq",
        "text": "DATSRAL (Data Structures and Algorithms Lecture) requires LBYCPEI as a hard prerequisite. LBYCPA2 (Data Structures and Algorithms Laboratory) requires DATSRAL as a co-requisite.",
        "metadata": {"term": "4", "type": "prerequisite", "course": "DATSRAL"}
    },
    {
        "id": "fundckt_prereq",
        "text": "FUNDCKT (Fundamentals of Electrical Circuits Lecture) requires ENGPHYS as a hard prerequisite. LBYEC2M (Fundamentals of Electrical Circuits Lab) requires FUNDCKT as a co-requisite.",
        "metadata": {"term": "4", "type": "prerequisite", "course": "FUNDCKT"}
    },
    # ── FIFTH TERM ────────────────────────────────────────────────────────────
    {
        "id": "term5_overview",
        "text": "TERM 5 - Fifth Term courses: ENGDATA (Engineering Data Analysis 1A, 3 units) requires CALENG2 or FNDSTAT as soft/hard prerequisite, NUMMETS (Numerical Methods 1E, 3 units) requires CALENG3 as hard prerequisite, FUNDLEC (Fundamentals of Electronic Circuits Lecture 1D, 3 units) requires FUNDCKT as hard prerequisite, LBYCPC2 (Fundamentals of Electronic Circuits Laboratory 1D, 1 unit) requires FUNDLEC as co-requisite, SOFDESG (Software Design Lecture 1E, 3 units) requires LBYCPA2 as hard prerequisite, LBYCPD2 (Software Design Laboratory 1E, 1 unit) requires SOFDESG as co-requisite, ENGENVI (Environmental Science and Engineering, 3 units) requires ENGCHEM as hard prerequisite, GEDANCE (Physical Fitness and Wellness in Dance 2C, 2 units), SAS2000 (Student Affairs Series 2, 0 units). Total: 19 units.",
        "metadata": {"term": "5", "type": "overview"}
    },
    {
        "id": "sofdesg_prereq",
        "text": "SOFDESG (Software Design Lecture) requires LBYCPA2 as a hard prerequisite. LBYCPD2 (Software Design Laboratory) requires SOFDESG as a co-requisite.",
        "metadata": {"term": "5", "type": "prerequisite", "course": "SOFDESG"}
    },
    {
        "id": "fundlec_prereq",
        "text": "FUNDLEC (Fundamentals of Electronic Circuits Lecture) requires FUNDCKT as a hard prerequisite. LBYCPC2 (Fundamentals of Electronic Circuits Laboratory) requires FUNDLEC as a co-requisite.",
        "metadata": {"term": "5", "type": "prerequisite", "course": "FUNDLEC"}
    },
    # ── SIXTH TERM ────────────────────────────────────────────────────────────
    {
        "id": "term6_overview",
        "text": "TERM 6 - Sixth Term courses: LCLSTWO (Lasallian Studies 2, 1 unit), LASARE2 (Lasallian Recollection 2, 0 units), MXSIGFN (Fundamentals of Mixed Signals and Sensors 1E, 3 units) requires FUNDLEC as hard prerequisite, LOGDSGN (Logic Circuits and Design Lecture 1E, 3 units) requires FUNDLEC as hard prerequisite, LBYCPG3 (Logic Circuits and Design Laboratory 1E, 1 unit) requires LOGDSGN as co-requisite, FDCNSYS (Feedback and Control Systems 1E, 3 units) requires NUMMETS as hard prerequisite, LBYCPC3 (Feedback and Control System Laboratory 1E, 1 unit) requires FDCNSYS as co-requisite, LBYME1C (Computer-Aided Drafting CAD for ECE and CpE 1C, 1 unit), GELACAH (Arts and Humanities 2B, 3 units), GESPORT (Physical Fitness and Wellness in Individual Sports 2C, 2 units). Total: 17 units.",
        "metadata": {"term": "6", "type": "overview"}
    },
    {
        "id": "logdsgn_prereq",
        "text": "LOGDSGN (Logic Circuits and Design Lecture) requires FUNDLEC as a hard prerequisite. LBYCPG3 (Logic Circuits and Design Laboratory) requires LOGDSGN as a co-requisite.",
        "metadata": {"term": "6", "type": "prerequisite", "course": "LOGDSGN"}
    },
    {
        "id": "fdcnsys_prereq",
        "text": "FDCNSYS (Feedback and Control Systems) requires NUMMETS as a hard prerequisite. LBYCPC3 (Feedback and Control System Laboratory) requires FDCNSYS as a co-requisite.",
        "metadata": {"term": "6", "type": "prerequisite", "course": "FDCNSYS"}
    },
    # ── SEVENTH TERM ──────────────────────────────────────────────────────────
    {
        "id": "term7_overview",
        "text": "TERM 7 - Seventh Term courses: GEETHIC (Ethics 2A, 3 units), MICPROS (Microprocessors Lecture 1E, 3 units) requires LOGDSGN as hard prerequisite, LBYCPA3 (Microprocessors Laboratory 1E, 1 unit) requires MICPROS as co-requisite, LBYCPB3 (Computer Engineering Drafting and Design Laboratory 1E, 1 unit) requires FUNDLEC and LOGDSGN as hard prerequisites, LBYEC3B (Intelligent Systems for Engineering, 1 unit) requires LBYEC2A and ENGDATA as hard prerequisites, LBYCPF2 (Introduction to HDL Laboratory 1E, 1 unit) requires FUNDLEC as hard prerequisite, DIGDACM (Data and Digital Communications 1E, 3 units) requires FUNDLEC as hard prerequisite, GETEAMS (Physical Fitness and Wellness in Team Sports 2C, 2 units), LBYCPG2 (Basic Computer Systems Administration, 1 unit). Total: 16 units.",
        "metadata": {"term": "7", "type": "overview"}
    },
    {
        "id": "MICPROS_prereq",
        "text": "MICPROS (Microprocessors Lecture) requires LOGDSGN as a hard prerequisite. LBYCPA3 (Microprocessors Laboratory) requires MICPROS as a co-requisite. MICPROS is taken in the Seventh Term.",
        "metadata": {"term": "7", "type": "prerequisite", "course": "MICPROS"}
    },
    # ── EIGHTH TERM ───────────────────────────────────────────────────────────
    {
        "id": "term8_overview",
        "text": "TERM 8 - Eighth Term courses: CSYSARC (Computer Architecture and Organization Lecture 1E, 3 units) requires MICPROS as hard prerequisite, LBYCPD3 (Computer Architecture and Organization Laboratory 1E, 1 unit) requires CSYSARC as co-requisite, EMBDSYS (Embedded Systems Lecture 1E, 3 units) requires MICPROS as hard prerequisite, LBYCPM3 (Embedded Systems Laboratory 1E, 1 unit) requires EMBDSYS as co-requisite, GELECST (Science and Technology 2B, 3 units), REMETHS (Methods of Research for CpE 1E, 3 units) requires ENGDATA, GEPCOMM, and LOGDSGN as hard prerequisites, OPESSYS (Operating Systems Lecture 1E, 3 units) requires LBYCPA2 as hard prerequisite, LBYCPO1 (Operating Systems Laboratory 1E, 1 unit) requires OPESSYS as co-requisite. Total: 8 units.",
        "metadata": {"term": "8", "type": "overview"}
    },
    {
        "id": "embdsys_prereq",
        "text": "EMBDSYS (Embedded Systems Lecture) requires MICPROS as a hard prerequisite. LBYCPM3 (Embedded Systems Laboratory) requires EMBDSYS as a co-requisite.",
        "metadata": {"term": "8", "type": "prerequisite", "course": "EMBDSYS"}
    },
    {
        "id": "csysarc_prereq",
        "text": "CSYSARC (Computer Architecture and Organization Lecture) requires MICPROS as a hard prerequisite. LBYCPD3 (Computer Architecture and Organization Laboratory) requires CSYSARC as a co-requisite.",
        "metadata": {"term": "8", "type": "prerequisite", "course": "CSYSARC"}
    },
    {
        "id": "remeths_prereq",
        "text": "REMETHS (Methods of Research for CpE) requires ENGDATA, GEPCOMM, and LOGDSGN as hard prerequisites.",
        "metadata": {"term": "8", "type": "prerequisite", "course": "REMETHS"}
    },
    # ── NINTH TERM ────────────────────────────────────────────────────────────
    {
        "id": "term9_overview",
        "text": "TERM 9 - Ninth Term courses: LCLSTRI (Lasallian Studies 3, 1 unit), LCASEAN (The Filipino and ASEAN, 3 units), LASARE3 (Lasallian Recollection 3, 0 units), DSIGPRO (Digital Signal Processing Lecture 1E, 3 units) requires FDCNSYS and EMBDSYS as hard/soft prerequisite, LBYCPA4 (Digital Signal Processing Laboratory 1E, 1 unit) requires DSIGPRO as co-requisite, OCHESAF (Basic Occupational Health and Safety 1E, 3 units) requires EMBDSYS as hard prerequisite, THSCP4A (CpE Practice and Design 1 1E, 1 unit) requires EMBDSYS and REMETHS as hard prerequisites, CPEPRAC (CpE Laws and Professional Practice 1E, 2 units) requires EMBDSYS as hard prerequisite, CPECOG1 (CpE Elective 1 Lecture 1F, 2 units) requires EMBDSYS and THSCP4A as hard/co prerequisite, LBYCPF3 (CpE Elective 1 Laboratory 1F, 1 unit) requires CPECOG1 as co-requisite. Total: 16 units.",
        "metadata": {"term": "9", "type": "overview"}
    },
    {
        "id": "thscp4a_prereq",
        "text": "THSCP4A (CpE Practice and Design 1) requires both EMBDSYS and REMETHS as hard prerequisites. This is a capstone/thesis preparation course taken in Term 9.",
        "metadata": {"term": "9", "type": "prerequisite", "course": "THSCP4A"}
    },
    {
        "id": "dsigpro_prereq",
        "text": "DSIGPRO (Digital Signal Processing Lecture) requires FDCNSYS as a hard prerequisite and EMBDSYS as a soft prerequisite. LBYCPA4 (Digital Signal Processing Laboratory) requires DSIGPRO as a co-requisite.",
        "metadata": {"term": "9", "type": "prerequisite", "course": "DSIGPRO"}
    },
    # ── TENTH TERM ────────────────────────────────────────────────────────────
    {
        "id": "term10_overview",
        "text": "TERM 10 - Tenth Term courses: LCENWRD (Encountering the Word in the World, 3 units), EMERTEC (Emerging Technologies in CpE 1E, 3 units) requires EMBDSYS as hard prerequisite, THSCP4B (CpE Practice and Design 2 1E, 1 unit) requires THSCP4A as hard prerequisite, ENGTREP (Technopreneurship 101 1C, 3 units) requires EMBDSYS as hard prerequisite, CONETSC (Computer Networks and Security Lecture 1E, 3 units) requires DIGDACM as hard prerequisite, LBYCPB4 (Computer Networks and Security Laboratory 1E, 1 unit) requires CONETSC as co-requisite, CPECAPS (Operational Technologies, 2 units) requires LBYCPB3 and LBYCPB4 as co/co requisite, CPECOG2 (CpE Elective 2 Lecture 1F, 2 units) requires THSCP4A as soft prerequisite, LBYCPH3 (CpE Elective 2 Laboratory 1F, 1 unit) requires CPECOG2 as co-requisite, SAS3000 (Student Affairs Series 3, 0 units) requires SAS2000 as hard prerequisite.",
        "metadata": {"term": "10", "type": "overview"}
    },
    {
        "id": "conetsc_prereq",
        "text": "CONETSC (Computer Networks and Security Lecture) requires DIGDACM as a hard prerequisite. LBYCPB4 (Computer Networks and Security Laboratory) requires CONETSC as a co-requisite.",
        "metadata": {"term": "10", "type": "prerequisite", "course": "CONETSC"}
    },
    {
        "id": "thscp4b_prereq",
        "text": "THSCP4B (CpE Practice and Design 2) requires THSCP4A as a hard prerequisite. This is the second part of the capstone/thesis sequence taken in Term 10.",
        "metadata": {"term": "10", "type": "prerequisite", "course": "THSCP4B"}
    },
    # ── ELEVENTH TERM ─────────────────────────────────────────────────────────
    {
        "id": "term11_overview",
        "text": "TERM 11 - Eleventh Term: PRCGECP (Practicum for CpE 1E, 3 units) requires REMETHS as hard prerequisite. Total: 3 units. This is the practicum/internship term.",
        "metadata": {"term": "11", "type": "overview"}
    },
    # ── TWELFTH TERM ──────────────────────────────────────────────────────────
    {
        "id": "term12_overview",
        "text": "TERM 12 - Twelfth Term courses: GERPHIS (Readings in the Philippine History 2A, 3 units), GEWORLD (The Contemporary World 2A, 3 units), THSCP4C (CpE Practice and Design 3 1E, 1 unit) requires THSCP4B as hard prerequisite, CPECOG3 (CpE Elective 3 Lecture 1F, 2 units) requires THSCP4A as soft prerequisite, LBYCPC4 (CpE Elective 3 Laboratory 1F, 1 unit) requires CPECOG3 as co-requisite, CPETRIP (Seminars and Field Trips for CpE 1E, 1 unit) requires EMBDSYS and CPECAPS as hard prerequisites, ECNOMIC (Engineering Economics for CpE 1C, 3 units) requires CALENG1 as soft prerequisite, ENGMANA (Engineering Management, 2 units) requires CALENG1 as soft prerequisite, GEUSELF (Understanding the Self 2A, 3 units). Total: 19 units.",
        "metadata": {"term": "12", "type": "overview"}
    },
    {
        "id": "thscp4c_prereq",
        "text": "THSCP4C (CpE Practice and Design 3) requires THSCP4B as a hard prerequisite. The full thesis sequence is THSCP4A (Term 9) then THSCP4B (Term 10) then THSCP4C (Term 12).",
        "metadata": {"term": "12", "type": "prerequisite", "course": "THSCP4C"}
    },
    # ── GENERAL POLICIES ──────────────────────────────────────────────────────
    {
        "id": "prereq_legend",
        "text": "Prerequisite Legend: H = Hard Pre-Requisite (must be passed before enrolling), S = Soft Pre-Requisite (Students may proceed even if they fail a soft prerequisite, as long as they enrolled in it. If they did not take it at all, the course will be INVALIDATED.), C = Co-Requisite (must be taken in the same term). This checklist is for freshmen who started AY 2022-2023.",
        "metadata": {"term": "all", "type": "policy"}
    },
    {
        "id": "checklist_warning",
        "text": "Students should not enroll without passing their respective hard prerequisites. Students may proceed even if they fail a soft prerequisite, as long as they enrolled in it. If they did not take it at all, the course will be INVALIDATED. This checklist is tentative and subject to change.",
        "metadata": {"term": "all", "type": "policy"}
    },
    {
        "id": "thesis_sequence",
        "text": "The CpE thesis/capstone sequence is: THSCP4A (Term 9, requires EMBDSYS and REMETHS) then THSCP4B (Term 10, requires THSCP4A) then THSCP4C (Term 12, requires THSCP4B). Students must complete this sequence to graduate.",
        "metadata": {"term": "all", "type": "policy"}
    },
    {
        "id": "nstp_sequence",
        "text": "The NSTP sequence is: NSTP101 (Term 1, General Orientation) then NSTPCW1 (Term 2) then NSTPCW2 (Term 3, requires NSTPCW1 as hard prerequisite).",
        "metadata": {"term": "all", "type": "policy"}
    },
    {
        "id": "lasallian_sequence",
        "text": "The Lasallian Studies sequence is: LCLSONE (Term 3) then LCLSTWO (Term 6) then LCLSTRI (Term 9). Lasallian Recollections: LASARE1 (Term 3), LASARE2 (Term 6), LASARE3 (Term 9).",
        "metadata": {"term": "all", "type": "policy"}
    },
    {
        "id": "sas_sequence",
        "text": "Student Affairs Series: SAS1000 (Term 3, 0 units) then SAS2000 (Term 5, 0 units) then SAS3000 (Term 10, 0 units, requires SAS2000 as hard prerequisite).",
        "metadata": {"term": "all", "type": "policy"}
    },
]


# OBJECTIVE 1: BENCHMARK TEST CASES WITH GROUND TRUTHS

TEST_CASES = [
    {
        "question": "What are the prerequisites for CALENG2?",
        "ground_truth": "CALENG2 requires CALENG1 as a hard prerequisite. Students must pass CALENG1 before enrolling in CALENG2."
    },
    {
        "question": "What are the prerequisites for DATSRAL?",
        "ground_truth": "DATSRAL requires LBYCPEI as a hard prerequisite."
    },
    {
        "question": "What is the co-requisite of LBYCPA2?",
        "ground_truth": "LBYCPA2 is the co-requisite of DATSRAL. Both must be taken in the same term."
    },
    {
        "question": "What courses are in the seventh term?",
        "ground_truth": "Seventh Term includes GEETHIC, MICPROS, LBYCPA3, LBYCPB3, LBYEC3B, LBYCPF2, DIGDACM, GETEAMS, and LBYCPG2. Total of 16 units."
    },
    {
        "question": "What does H mean in the prerequisite legend?",
        "ground_truth": "H means Hard Pre-Requisite. Students must pass the hard prerequisite course before enrolling in the next course."
    },
    {
        "question": "What happens if I take a course without passing its soft prerequisite?",
        "ground_truth": "Students may still proceed to the next course even if they fail a soft prerequisite, as long as they enrolled in it. However, if a student does not take the soft prerequisite course at all, the subsequent course will be INVALIDATED."
    },
    {
        "question": "What are the prerequisites for THSCP4A?",
        "ground_truth": "THSCP4A requires both EMBDSYS and REMETHS as hard prerequisites."
    },
    {
        "question": "What is the thesis sequence for CpE students?",
        "ground_truth": "The thesis sequence is THSCP4A in Term 9, then THSCP4B in Term 10, then THSCP4C in Term 12."
    },
    {
        "question": "What are the prerequisites for LOGDSGN?",
        "ground_truth": "LOGDSGN requires FUNDLEC as a hard prerequisite."
    },
    {
        "question": "What are the prerequisites for EMBDSYS?",
        "ground_truth": "EMBDSYS requires MICPROS as a hard prerequisite."
    },
    {
        "question": "What is the prerequisite for MICPROS?",
        "ground_truth": "MICPROS requires LOGDSGN as a hard prerequisite."
    },
    {
        "question": "What term is PRCGECP taken and what is its prerequisite?",
        "ground_truth": "PRCGECP is taken in the Eleventh Term and requires REMETHS as a hard prerequisite."
    },
    {
        "question": "What is the co-requisite relationship between SOFDESG and LBYCPD2?",
        "ground_truth": "LBYCPD2 is the laboratory co-requisite of SOFDESG. Both must be taken in the same term."
    },
    {
        "question": "What prerequisites does REMETHS require?",
        "ground_truth": "REMETHS requires ENGDATA, GEPCOMM, and LOGDSGN as hard prerequisites."
    },
    {
        "question": "Can I take FUNDLEC without passing FUNDCKT?",
        "ground_truth": "No. FUNDLEC requires FUNDCKT as a hard prerequisite and must be passed before enrolling."
    },
]


# OBJECTIVE 4: MODELS TO COMPARE (enable/disable by commenting out)

MODELS_TO_TEST = [
    #  Hugging Face 
    {
        "provider": "hf",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "label":    "Llama-3.1-8B"
    },
     {
         "provider": "hf",
         "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
         "label":    "DeepSeek-R1-8B"
     },
     {
         "provider": "hf",
         "model_id": "Qwen/Qwen2.5-7B-Instruct",
         "label":    "Qwen2.5-7B"
     },
    {
        "provider": "hf",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "label":    "Mistral-7B-v0.2"
    },
    # Google Gemini (needs GEMINI_API_KEY in .env) 
    {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash-lite",
        "label":    "Gemini-2.5-Flash-Lite"
    },

    # OpenAI (needs OPENAI_API_KEY in .env) 
     {
         "provider": "openai",
         "model_id": "gpt-4o-mini",
         "label":    "GPT-4o-mini"
     },
]

# OBJECTIVE 5: PARAMETER CONFIGS TO TUNE 
PARAM_CONFIGS = [
    {"temperature": 0.1, "max_tokens": 200, "label": "t=0.1 tok=200"},
    {"temperature": 0.0, "max_tokens": 200, "label": "t=0.0 tok=200"},
    {"temperature": 0.3, "max_tokens": 200, "label": "t=0.3 tok=200"},
    {"temperature": 0.1, "max_tokens": 400, "label": "t=0.1 tok=400"},
]



# CHROMADB — build once, reused across all models

chroma_client = chromadb.Client()

def build_vector_store() -> chromadb.Collection:
    print("Building ChromaDB vector store with all-MiniLM-L6-v2...")
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    try:
        chroma_client.delete_collection(name="dlsu_cpe_checklist")
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name="dlsu_cpe_checklist",
        embedding_function=embedding_fn,
    )
    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        metadatas=[doc["metadata"] for doc in DOCUMENTS],
    )
    print(f"Stored {len(DOCUMENTS)} document chunks\n")
    return collection


def retrieve(collection: chromadb.Collection, query: str, top_k: int = 3) -> str:
    results = collection.query(query_texts=[query], n_results=top_k)
    return "\n\n".join(results["documents"][0])


# GENERATION — supports HF, Gemini, OpenAI

SYSTEM_PROMPT = (
    "You are an academic adviser for DLSU Computer Engineering students. "
    "Use ONLY the provided context to answer questions accurately. "
    "If the answer is not in the context, say: "
    "'I don't have that information — please consult your adviser.'"
)

def generate(client: dict, model_id: str, context: str,
             question: str, config: dict) -> str:

    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    #  Hugging Face 
    if client["provider"] == "hf":
        response = client["instance"].chat_completion(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        answer = response.choices[0].message.content.strip()

    #  Google Gemini 
    elif client["provider"] == "gemini":
        from google import genai as google_genai
        from google.genai import types

        gemini_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        response = gemini_client.models.generate_content(
            model=model_id,
            contents=f"{SYSTEM_PROMPT}\n\n{user_msg}",
            config=types.GenerateContentConfig(
                max_output_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
        )
        answer = response.text.strip()

    #  OpenAI 
    elif client["provider"] == "openai":
        response = client["instance"].chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        answer = response.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown provider: {client['provider']}")

    # Strip DeepSeek thinking blocks 
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

    return answer



# OBJECTIVE 2: SCORING — BLEU, METEOR, ROUGE-1, ROUGE-L, BERTScore
 
rouge_eval = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smoother   = SmoothingFunction().method4
 
def score_response(ground_truth: str, answer: str) -> dict:
    # ROUGE
    r    = rouge_eval.score(ground_truth, answer)
    # BLEU
    ref  = [ground_truth.lower().split()]
    hyp  = answer.lower().split()
    bleu = sentence_bleu(ref, hyp, smoothing_function=smoother)
    # METEOR
    met  = meteor_score([ground_truth.lower().split()], answer.lower().split())
    # BERTScore — uses contextual embeddings (roberta-large)
    # Returns precision, recall, F1 — we use F1 as the reported score
    P, R, F1 = bert_score(
        [answer],
        [ground_truth],
        lang="en",
        model_type="roberta-large",
        verbose=False,
    )
    bscore = F1[0].item()
 
    return {
        "rouge1":     r['rouge1'].fmeasure,
        "rouge_l":    r['rougeL'].fmeasure,
        "bleu":       bleu,
        "meteor":     met,
        "bert_score": bscore,
    }



# OBJECTIVE 3: HALLUCINATION DETECTION

def detect_hallucination(ground_truth: str, answer: str) -> tuple:
    gt  = ground_truth.lower()
    ans = answer.lower()

    # Check 1: retrieval miss
    no_info = ["don't have that information", "cannot find", "no information"]
    if any(p in ans for p in no_info) and len(ground_truth) > 30:
        return True, "RETRIEVAL MISS"

    # Check 2: contradiction — answer says yes when ground truth says no
    gt_negative   = any(w in gt  for w in ["no.", "cannot", "must not", "not allowed"])
    ans_positive  = any(w in ans for w in ["yes,", "yes.", "you can", "is allowed"])
    ans_agrees_no = any(w in ans for w in ["no,", "no.", "cannot", "must not"])
    if gt_negative and ans_positive and not ans_agrees_no:
        return True, "CONTRADICTION"

    # Check 3: wrong course codes in answer
    gt_codes  = set(re.findall(r'\b[A-Z]{3,8}\d*[A-Z]?\b', ground_truth))
    ans_codes = set(re.findall(r'\b[A-Z]{3,8}\d*[A-Z]?\b', answer))
    noise     = {"I", "H", "S", "C", "OK", "ONLY", "DLSU", "NO", "YES",
                 "BOTH", "TERM", "CpE", "AY", "NOTE"}
    wrong     = ans_codes - gt_codes - noise
    if wrong:
        return True, f"WRONG CODES: {wrong}"

    return False, "OK"


# EVALUATION LOOP
 
def run_evaluation(model_label: str, model_id: str, config: dict,
                   collection: chromadb.Collection, client: dict) -> dict:
 
    print(f"\n  --- Config: {config['label']} ---")
 
    all_scores       = []
    retrieval_times  = []
    generation_times = []
    total_times      = []
    halluc_flags     = []
    results_log      = []
 
    for i, test in enumerate(TEST_CASES):
        total_start = time.time()
 
        t0       = time.time()
        context  = retrieve(collection, test["question"], top_k=3)
        ret_time = time.time() - t0
 
        t1       = time.time()
        answer   = generate(client, model_id, context, test["question"], config)
        gen_time = time.time() - t1
 
        total_time = time.time() - total_start
 
        scores               = score_response(test["ground_truth"], answer)
        is_halluc, h_reason  = detect_hallucination(test["ground_truth"], answer)
 
        all_scores.append(scores)
        retrieval_times.append(ret_time)
        generation_times.append(gen_time)
        total_times.append(total_time)
        halluc_flags.append(is_halluc)
 
        h_tag = "[HALLUC]" if is_halluc else "[OK]"
        t_tag = "[OK]" if total_time < 5 else "[OVER 5s]"
 
        print(f"  Q{i+1:02d}: {test['question'][:55]}")
        print(f"        Answer:  {answer[:100]}...")
        print(f"        ROUGE-1:{scores['rouge1']:6.3f} | ROUGE-L:{scores['rouge_l']:6.3f} | "
              f"BLEU:{scores['bleu']:6.3f} | METEOR:{scores['meteor']:6.3f} | "
              f"BERTScore:{scores['bert_score']:6.3f}")
        print(f"        Time:{total_time:5.2f}s {t_tag} | Hallucination: {h_tag} {h_reason}")
        print()
 
        results_log.append({
            "question":        test["question"],
            "ground_truth":    test["ground_truth"],
            "answer":          answer,
            **scores,
            "hallucination":   is_halluc,
            "halluc_reason":   h_reason,
            "retrieval_time":  ret_time,
            "generation_time": gen_time,
            "total_time":      total_time,
        })
    n            = len(TEST_CASES)
    halluc_count = sum(halluc_flags)
    def avg(key): return sum(s[key] for s in all_scores) / n
 
    summary = {
        "model":               model_label,
        "provider":            client["provider"],
        "config":              config["label"],
        "avg_rouge1":          avg("rouge1"),
        "avg_rouge_l":         avg("rouge_l"),
        "avg_bleu":            avg("bleu"),
        "avg_meteor":          avg("meteor"),
        "avg_bert_score":      avg("bert_score"),
        "hallucination_count": halluc_count,
        "hallucination_rate":  halluc_count / n * 100,
        "avg_retrieval_time":  sum(retrieval_times)  / n,
        "avg_generation_time": sum(generation_times) / n,
        "avg_total_time":      sum(total_times)      / n,
        "pct_under_5s":        sum(1 for t in total_times if t < 5) / n * 100,
        "detail":              results_log,
    }
 
    print(f"  SUMMARY [{config['label']}]")
    print(f"  ROUGE-1:{summary['avg_rouge1']:6.3f} | ROUGE-L:{summary['avg_rouge_l']:6.3f} | "
          f"BLEU:{summary['avg_bleu']:6.3f} | METEOR:{summary['avg_meteor']:6.3f} | "
          f"BERTScore:{summary['avg_bert_score']:6.3f}")
    print(f"  Hallucinations: {halluc_count}/{n} ({summary['hallucination_rate']:.0f}%)")
    print(f"  Avg Time: {summary['avg_total_time']:.2f}s | Under 5s: {summary['pct_under_5s']:.0f}%")
 
    return summary



# MAIN

if __name__ == "__main__":

    clients = {}

    if os.getenv("HF_TOKEN"):
        clients["hf"] = {
            "provider": "hf",
            "instance": InferenceClient(token=os.getenv("HF_TOKEN"))
        }

    if os.getenv("GEMINI_API_KEY"):
        try:
            from google import genai as google_genai
            clients["gemini"] = {"provider": "gemini", "instance": None}
            print("Gemini client initialized")
        except ImportError:
            print("[SKIP] Gemini — run: pip install google-genai")

    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            clients["openai"] = {
                "provider": "openai",
                "instance": OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            }
            print("OpenAI client initialized")
        except ImportError:
            print("[SKIP] OpenAI — run: pip install openai")

    collection = build_vector_store()

    all_summaries = []

    for model_config in MODELS_TO_TEST:
        provider  = model_config["provider"]
        model_id  = model_config["model_id"]
        label     = model_config["label"]

        if provider not in clients:
            print(f"\n[SKIP] {label} — {provider} client not initialized "
                  f"(check your .env for the required API key)")
            continue

        client = clients[provider]

        print(f"\n{'='*65}")
        print(f"MODEL: {label}  |  Provider: {provider}")
        print(f"{'='*65}")

        for param_config in PARAM_CONFIGS:
            try:
                summary = run_evaluation(
                    label, model_id, param_config, collection, client
                )
                all_summaries.append(summary)
            except Exception as e:
                print(f"  [SKIPPED] {param_config['label']} — {e}\n")
                continue

    #Final comparison table 
    if all_summaries:
        print(f"\n{'='*100}")
        print("FINAL COMPARISON TABLE")
        print(f"{'='*100}")
        print(f"{'Model + Config':<42} {'R-1':>6} {'R-L':>6} {'BLEU':>6} "
              f"{'METEOR':>7} {'BERT':>7} {'Halluc':>8} {'AvgTime':>8} {'<5s':>5}")
        print("-" * 110)

        for s in sorted(all_summaries, key=lambda x: -x["avg_rouge_l"]):
            name = f"{s['model']} [{s['config']}]"
            print(
                f"{name:<42} "
                f"{s['avg_rouge1']:>6.3f} "
                f"{s['avg_rouge_l']:>6.3f} "
                f"{s['avg_bleu']:>6.3f} "
                f"{s['avg_meteor']:>7.3f} "
                f"{s['avg_bert_score']:>7.3f} "
                f"{s['hallucination_count']:>4}/{len(TEST_CASES)} "
                f"{s['avg_total_time']:>7.2f}s "
                f"{s['pct_under_5s']:>4.0f}%"
            )

        # Save full results for thesis records
        with open("baseline_results.json", "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to baseline_results.json")

    else:
        print("\nNo results collected — check your API keys and model list.")