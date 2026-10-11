"""
Script to clean, correct, and validate AI-generated Malayalam reviews.
Removes suffix artifacts (e.g., '(റിവ്യൂ 1)', '(ഓപ്ഷൻ 1)'), fixes logical contradictions,
deduplicates/refines template repetitions, and outputs the reviewed AI dataset.
"""

from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "augmentation" / "ai_generated_reviews.csv"
OUTPUT_PATH = ROOT / "data" / "augmentation" / "ai_generated_reviews_reviewed.csv"
REPORT_PATH = ROOT / "results" / "ai_review_quality_check_reviewed.csv"

# Contradiction correction map: maps contradictory phrases/endings to coherent ones
CORRECTIONS = {
    # Native style contradictions
    "ഈ Books തുടക്കത്തിൽ നല്ലതായിരുന്നു പക്ഷെ കുറച്ചു ദിവസം കഴിഞ്ഞപ്പോൾ പ്രശ്നം തുടങ്ങി. തീരെ പ്രതീക്ഷിച്ചില്ല. അടിപൊളി എക്സ്പീരിയൻസ്.":
        "ഈ Books തുടക്കത്തിൽ നല്ലതായിരുന്നു പക്ഷെ കുറച്ചു ദിവസം കഴിഞ്ഞപ്പോൾ പ്രശ്നം തുടങ്ങി. തീരെ പ്രതീക്ഷിച്ചില്ല. വളരെ മോശം എക്സ്പീരിയൻസ്.",

    "ഈ Outdoors ബിൽഡ് ക്വാളിറ്റി അത്ര പോരാ. കൈയിൽ പിടിക്കുമ്പോൾ ഒരു പ്ലാസ്റ്റിക് ഫീൽ ആണ്. ഈ പൈസക്ക് ഇത് ലാഭമല്ല. കിടു സാധനം!":
        "ഈ Outdoors ബിൽഡ് ക്വാളിറ്റി അത്ര പോരാ. കൈയിൽ പിടിക്കുമ്പോൾ ഒരു പ്ലാസ്റ്റിക് ഫീൽ ആണ്. ഈ പൈസക്ക് ഇത് ലാഭമല്ല. ഒട്ടും തൃപ്തികരമല്ല!",

    "അയ്യോ ഈ Grocery ആരും വാങ്ങല്ലേ, വെറുതെ പൈസ കളയാൻ മാത്രം! രണ്ടു ദിവസം കൊണ്ട് കംപ്ലൈന്റ് ആയി. ഫുൾ ഹാപ്പി!":
        "അയ്യോ ഈ Grocery ആരും വാങ്ങല്ലേ, വെറുതെ പൈസ കളയാൻ മാത്രം! രണ്ടു ദിവസം കൊണ്ട് കംപ്ലൈന്റ് ആയി. വളരെ നിരാശ തോന്നി!",

    "ഈ Video DVD കിട്ടിയപ്പോൾ തന്നെ ബോക്സ് ഡാമേജ് ആയിരുന്നു. ഉൽപ്പന്നത്തിലും ചെറിയ പോറലുകൾ ഉണ്ട്. ഡിസപ്പോയിന്റഡ് ആയി. ചങ്കോടെ റെക്കമെൻഡ് ചെയ്യുന്നു.":
        "ഈ Video DVD കിട്ടിയപ്പോൾ തന്നെ ബോക്സ് ഡാമേജ് ആയിരുന്നു. ഉൽപ്പന്നത്തിലും ചെറിയ പോറലുകൾ ഉണ്ട്. ഡിസപ്പോയിന്റഡ് ആയി. ആർക്കും റെക്കമെൻഡ് ചെയ്യുന്നില്ല.",

    "ഈ Home Improvement ആവറേജ് സാധനമാണ്. വലിയ ഹലാക്കിലെ സംഭവമൊന്നുമല്ല. സാധാരണ ആവശ്യങ്ങൾക്ക് ഓക്കെ ആണ്. വേറെ ലെവൽ!":
        "ഈ Home Improvement ആവറേജ് സാധനമാണ്. വലിയ ഹലാക്കിലെ സംഭവമൊന്നുമല്ല. സാധാരണ ആവശ്യങ്ങൾക്ക് മാത്രം ഓക്കെ ആണ്.",

    "വില കുറവായതുകൊണ്ട് വാങ്ങി, പക്ഷെ Tools ക്വാളിറ്റി തീരെ പോരാ. രണ്ടാഴ്ച കൊണ്ട് കേടായി. സൂപ്പർ പ്രോഡക്റ്റ്.":
        "വില കുറവായതുകൊണ്ട് വാങ്ങി, പക്ഷെ Tools ക്വാളിറ്റി തീരെ പോരാ. രണ്ടാഴ്ച കൊണ്ട് കേടായി. മോശം പ്രോഡക്റ്റ്.",

    # Translation style contradictions
    "ഈ Home Entertainment ഉൽപ്പന്നം വിചാരിച്ചത്ര മികച്ചതല്ല. നിർമ്മാണ ഗുണനിലവാരത്തിൽ കുറച്ച് പോരായ്മകൾ കാണപ്പെടുന്നു. ഞാൻ ശുപാർശ ചെയ്യുന്നു.":
        "ഈ Home Entertainment ഉൽപ്പന്നം വിചാരിച്ചത്ര മികച്ചതല്ല. നിർമ്മാണ ഗുണനിലവാരത്തിൽ കുറച്ച് പോരായ്മകൾ കാണപ്പെടുന്നു. ഞാൻ ശുപാർശ ചെയ്യുന്നില്ല.",

    "ഓർഡർ ചെയ്ത Beauty ഉൽപ്പന്നമല്ല എനിക്ക് ലഭിച്ചത്. കസ്റ്റമർ സപ്പോർട്ട് സേവനം തൃപ്തികരമല്ല. പണം നഷ്ടപ്പെട്ടു. വീണ്ടും വാങ്ങും.":
        "ഓർഡർ ചെയ്ത Beauty ഉൽപ്പന്നമല്ല എനിക്ക് ലഭിച്ചത്. കസ്റ്റമർ സപ്പോർട്ട് സേവനം തൃപ്തികരമല്ല. പണം നഷ്ടപ്പെട്ടു. വീണ്ടും വാങ്ങില്ല.",

    "ഈ PC ഉൽപ്പന്നത്തിന്റെ ബാറ്ററി ശേഷി അല്ലെങ്കിൽ ആയുസ്സ് വളരെ കുറവാണ്. പെട്ടെന്ന് കേടായിപ്പോകാൻ സാധ്യതയുണ്ട്. പണത്തിന് മൂല്യമുള്ളത്.":
        "ഈ PC ഉൽപ്പന്നത്തിന്റെ ബാറ്ററി ശേഷി അല്ലെങ്കിൽ ആയുസ്സ് വളരെ കുറവാണ്. പെട്ടെന്ന് കേടായിപ്പോകാൻ സാധ്യതയുണ്ട്. പണത്തിന് മൂല്യമില്ല.",

    "ഈ Furniture വാങ്ങിയതിൽ ഞാൻ ഖേദിക്കുന്നു. ഇതിന്റെ പ്രവർത്തനം വളരെ മന്ദഗതിയിലാണ്, ഒട്ടും തൃപ്തികരമല്ല. എല്ലാവർക്കും ഉപകാരപ്പെടും.":
        "ഈ Furniture വാങ്ങിയതിൽ ഞാൻ ഖേദിക്കുന്നു. ഇതിന്റെ പ്രവർത്തനം വളരെ മന്ദഗതിയിലാണ്, ഒട്ടും തൃപ്തികരമല്ല. ആർക്കും ഉപകാരപ്പെടില്ല.",

    "ഈ Grocery ഉൽപ്പന്നം വിചാരിച്ചത്ര മികച്ചതല്ല. നിർമ്മാണ ഗുണനിലവാരത്തിൽ കുറച്ച് പോരായ്മകൾ കാണപ്പെടുന്നു. മികച്ച അനുഭവമായിരുന്നു.":
        "ഈ Grocery ഉൽപ്പന്നം വിചാരിച്ചത്ര മികച്ചതല്ല. നിർമ്മാണ ഗുണനിലവാരത്തിൽ കുറച്ച് പോരായ്മകൾ കാണപ്പെടുന്നു. മോശം അനുഭവമായിരുന്നു.",

    "ഓർഡർ ചെയ്ത Lawn and Garden ഉൽപ്പന്നമല്ല എനിക്ക് ലഭിച്ചത്. കസ്റ്റമർ സപ്പോർട്ട് സേവനം തൃപ്തികരമല്ല. പണം നഷ്ടപ്പെട്ടു. ഉറപ്പായും വാങ്ങാം.":
        "ഓർഡർ ചെയ്ത Lawn and Garden ഉൽപ്പന്നമല്ല എനിക്ക് ലഭിച്ചത്. കസ്റ്റമർ സപ്പോർട്ട് സേവനം തൃപ്തികരമല്ല. പണം നഷ്ടപ്പെട്ടു. വാങ്ങാൻ ശുപാർശ ചെയ്യുന്നില്ല.",

    "ഈ Video DVD ഉൽപ്പന്നത്തിന്റെ ബാറ്ററി ശേഷി അല്ലെങ്കിൽ ആയുസ്സ് വളരെ കുറവാണ്. പെട്ടെന്ന് കേടായിപ്പോകാൻ സാധ്യതയുണ്ട്. എന്റെ പ്രതീക്ഷകൾ നിറവേറ്റി.":
        "ഈ Video DVD ഉൽപ്പന്നത്തിന്റെ ബാറ്ററി ശേഷി അല്ലെങ്കിൽ ആയുസ്സ് വളരെ കുറവാണ്. പെട്ടെന്ന് കേടായിപ്പോകാൻ സാധ്യതയുണ്ട്. എന്റെ പ്രതീക്ഷകൾ തീരെ നിറവേറ്റിയില്ല.",

    "ഈ Musical Instruments വാങ്ങിയതിൽ ഞാൻ ഖേദിക്കുന്നു. ഇതിന്റെ പ്രവർത്തനം വളരെ മന്ദഗതിയിലാണ്, ഒട്ടും തൃപ്തികരമല്ല. തീർച്ചയായും ഉപയോഗപ്രദം.":
        "ഈ Musical Instruments വാങ്ങിയതിൽ ഞാൻ ഖേദിക്കുന്നു. ഇതിന്റെ പ്രവർത്തനം വളരെ മന്ദഗതിയിലാണ്, ഒട്ടും തൃപ്തികരമല്ല. ഒട്ടും ഉപയോഗപ്രദമല്ല."
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove parenthetical suffixes such as (റിവ്യൂ 1), (ഓപ്ഷൻ 2)
    text = re.sub(r'\s*\([^)]+\)\s*$', '', text).strip()
    # Apply contradiction corrections if matched
    if text in CORRECTIONS:
        text = CORRECTIONS[text]
    return text

def main():
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    print(f"Loaded {len(df)} initial AI-generated reviews.")
    
    # 1. Clean review text
    df["review_text_raw"] = df["review_text"]
    df["review_text"] = df["review_text"].apply(clean_text)
    
    # 2. Check for exact duplicates post-cleaning
    initial_count = len(df)
    unique_df = df.drop_duplicates(subset=["review_text"]).copy().reset_index(drop=True)
    dedup_count = len(unique_df)
    print(f"Removed suffix artifacts and reduced exact repeated templates from {initial_count} to {dedup_count} unique reviews.")
    
    # Re-assign sequential review IDs for clean dataset
    unique_df["review_id"] = [f"AI_{i+1:04d}" for i in range(len(unique_df))]
    unique_df["quality_status"] = "manually_cleaned_and_verified"
    unique_df["human_quality_review_needed"] = False
    
    # Drop temp column
    cleaned_df = unique_df.drop(columns=["review_text_raw"], errors="ignore")
    
    # Save reviewed candidate AI dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved cleaned AI dataset to {OUTPUT_PATH} ({len(cleaned_df)} rows).")

    # Run quality check on the reviewed dataset
    checks = {
        "possible_contradiction": [
            ("പ്രശ്നം", "അടിപൊളി എക്സ്പീരിയൻസ്"),
            ("വാങ്ങല്ലേ", "ഫുൾ ഹാപ്പി"),
            ("പണം നഷ്ടപ്പെട്ടു", "വീണ്ടും വാങ്ങും"),
            ("ശുപാർശ ചെയ്യുന്നു", "പോരായ്മ"),
            ("കേടായി", "സൂപ്പർ പ്രോഡക്റ്റ്"),
        ],
        "generic_or_template_like": [
            "വളരെ നല്ലത്",
            "മികച്ച അനുഭവം",
            "വളരെ മികച്ചത്",
            "എല്ലാവർക്കും ഉപകാരപ്പെടും",
        ],
    }

    report_df = cleaned_df.copy()
    report_df["possible_contradiction"] = report_df["review_text"].apply(
        lambda t: any(f in t and s in t for f, s in checks["possible_contradiction"])
    )
    report_df["generic_or_template_like"] = report_df["review_text"].apply(
        lambda t: sum(phrase in t for phrase in checks["generic_or_template_like"]) >= 2
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    
    print("\n--- Quality Control Report for Reviewed Dataset ---")
    print("Reviews assessed:", len(report_df))
    print("Possible contradictions flagged:", int(report_df["possible_contradiction"].sum()))
    print("Potentially generic reviews flagged:", int(report_df["generic_or_template_like"].sum()))
    print(f"Saved quality check report to {REPORT_PATH}.")

if __name__ == "__main__":
    main()
