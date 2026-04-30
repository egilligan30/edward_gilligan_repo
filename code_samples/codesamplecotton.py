"""
Edward Gilligan -- Python Coding Sample

Part of the larger cottonclean.py
=============================================================
Cleans historical cotton industrial revolution data using Polars.

Three cleaning tasks:
  1. Forward-fill Town / District across blank rows.
  2. Contact info that ended up in Firm/Owner Name gets moved to
     Pay Day / Notes and saved in an audit column so nothing is lost.
  3. Weft / Twist columns are checked against a free-text Details field —
     fixes swaps, fills in missing values, and flags entries that look made up.

On the source data:
  The source is Worrall's Cotton Spinners' & Manufacturers'
  Directory, a late Victorian / Edwardian trade directory of
  cotton firms. The pages were photocopied, scanned to PDF, and then
  imputed into a structured Excel format by Claude using
  a standardised prompt applied to every set of four pages. The three
  cleaning tasks here correct systematic imputation errors that arise from
  the mismatch between Worrall's layout and a fixed extraction template.

  In Worrall's, each firm's entry is a dense prose block — firm name, mill
  name, spindle and loom counts, weft and twist counts, products, pay day,
  telegram address, and telephone number all run together in a single
  paragraph with no column separators. Locations are not repeated per
  entry; instead, a bold town header (e.g. BACUP, BLACKBURN) covers a
  block of firms, and sub-location headers (e.g. BRITANNIA, STACKSTEADS)
  cover smaller groups within that block. Claude imputed row by row against
  a fixed column template, which produces three predictable error types:
  location cells go blank after the first row of each block; contact
  details that appear at the end of a prose entry get absorbed into the
  firm name field; and weft / twist counts, which Worrall's gives inline
  as e.g. "15s/24s weft, 18s/20s twist", are sometimes swapped or only
  partially extracted into the two structured columns. None of these are
  treated as data loss; the goal is to recover intent while keeping an
  auditable record of every change made.

Note on map_rows():
  Fixes 2 and 3 need to look at multiple columns at once and make decisions
  based on what's in them which isn't something a single Polars expression
  can do cleanly. So map_rows() is used on purpose. I trade a bit of speed
  for correct logic. On a big dataset would probably be best to rewrite
  this as a vectorised expression.

Dependencies: polars, re
"""

import polars as pl
import re


# =============================================================================
# KEYWORD LISTS
# These are used by the functions below to figure out what kind of content
# is sitting in a cell.
# The terms reflect how contact information appears in Worrall's directory
# entries. Telegram addresses follow the pattern 'Telegrams, "Alias, Town"'
# and telephone numbers appear as 'Telephone No. X' or 'Telephone Nos.—
# Mill A, X; Mill B, Y' for firms with multiple sites. Both appear at the
# end of the prose entry block, immediately after pay day information, which
# is why Claude's fixed-column prompt sometimes pulled them into the firm
# name field rather than leaving them in Pay Day / Notes.
# =============================================================================

CONTACT_WORDS = [
    "telephone", "telegrams", "telegram", "tel", "phone",
    "t.n.", "t.a.", "national telephone", "post office",
]


# =============================================================================
# HELPER PREDICATES & STRING UTILITIES
# =============================================================================

def looks_like_contact_info(text: str) -> bool:
    """
    Returns True if the text looks like a phone or telegram contact.
    The short-string-with-number heuristic catches bare telephone numbers
    that Claude extracted without the 'Telephone No.' label. In Worrall's
    these are typically 2–4 digit local exchange numbers, short enough that
    a number-only string under 60 characters is almost certainly not a firm
    name or product description.
    """
    if not text:
        return False
    t = text.lower().strip()
    if any(word in t for word in CONTACT_WORDS):
        return True
    # a short string with a 3+ digit number is probably a phone/reference number
    if re.search(r'\b(no\.?\s*)?\d{3,}\b', t) and len(t) < 60:
        return True
    return False


def append_note(existing: str | None, addition: str) -> str:
    """
    Tacks addition onto existing, separated by a semicolon.
    The strip(";") handles edge cases where existing already ends with a
    semicolon. This happens when a row had partially cleaned notes from
    an earlier pass.
    """
    existing = existing.strip().strip(";").strip() if existing else ""
    addition = addition.strip()
    return (existing + "; " + addition).strip("; ") if existing else addition


def send_to_extra(current_extra: str | None, value: str, source_col: str) -> str:
    """
    Adds a displaced value to the audit column, tagged with where it came from.
    Returns a new string instead of changing the existing one. This keeps
    things consistent with how Polars works, since it never edits data in place.
    The source_col tag matters for verification. Worrall's entries are
    recoverable from the original PDFs, but going back to a scanned photocopy
    is slow. The tag lets you identify and manually review any case where the
    heuristic may have misfired without leaving the main columns in an
    ambiguous state.
    """
    value = value.strip()
    if not value:
        return current_extra or ""
    tag      = f"[from {source_col}] {value}"
    existing = (current_extra or "").strip(" | ")
    return (existing + " | " + tag).strip(" | ") if existing else tag


EXTRA_COL = "extra_updated"   # name of the audit column


# =============================================================================
# FIX 1 — FORWARD-FILL LOCATION
# Town / District is usually only written once for a specific group of rows. This
# copies it down into the blank rows below so every row has a value.
# forward_fill() does exactly that. It takes the last non-null value and
# repeats it downward until it hits another non-null value.
# with_columns() swaps in the updated column and leaves everything else alone.
# The function takes and returns a LazyFrame so nothing actually runs until
# .collect() is called later, i.e., the point of lazy evaluation.
#
# On the data:
# In Worrall's, town and sub-location names are printed as bold section
# headers — BACUP, then BRITANNIA and STACKSTEADS beneath it — covering
# all firm entries that follow until the next header. Claude's prompt
# extracted entries row by row against a fixed template, so it correctly
# populated Town / District for the first row under each header but left
# it blank for every subsequent row in that block. The blanks are therefore
# structural: they mean "same location as the row above", not "location
# unknown". Forward-filling recovers that without making any assumption
# about the data that isn't already explicit in the original directory layout.
# =============================================================================

def forward_fill_location(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Copies Town / District down into blank rows below it."""
    return lf.with_columns(
        pl.col("Town / District").forward_fill()
    )


# =============================================================================
# FIX 2 — CONTACT INFO IN FIRM/OWNER NAME
# Sometimes a phone or telegram number got entered into the firm name field.
# This splits that field on semicolons and newlines, pulls out any contact
# fragments, moves them to the Notes column, and logs them in the audit column.
# map_rows() used because this fix needs to read one column, make a decision,
# then write to two other columns all in the same pass. A single Polars
# expression can't do that, so drop into Python row-by-row logic instead.
# Steps:
#   1. lf.collect()       — turn the LazyFrame into a real df so we can loop over it
#   2. df.select(cols)    — only grab the columns we're actually changing
#   3. .map_rows(fn)      — run our Python function on each row; gives back a df
#                           with columns named column_0, column_1, etc.
#   4. .rename(...)       — put the real column names back using their positions
#   5. with_columns(...)  — write the fixed columns back into the full df
#   6. df.lazy()          — wrap it back up as a LazyFrame for the next step
#
# On the data:
# In Worrall's, a firm's telephone number and telegram address appear at
# the end of its prose entry block, immediately after pay day information
# and sometimes after a second address or warehouse location. Because
# Claude's extraction prompt used a fixed column template rather than parsing
# the prose sequentially, it occasionally ran contact details into the firm
# name field. This most likely occured when a long entry caused the model to lose track
# of field boundaries mid-paragraph. The pattern is consistent: telegram
# addresses follow 'Telegrams, "Alias, Town."' and telephone numbers follow
# 'Telephone No. X', so the keyword list catches them reliably. Contact
# details are moved to Pay Day / Notes rather than dropped because telegram
# aliases and early telephone exchange numbers are useful for cross-
# referencing firms across different years/editions of Worrall's or against
# contemporary Post Office directories.
# =============================================================================

def _fix_contact_firm_row(row: dict) -> dict:
    """
    The actual row-level logic. Gets a dict of {column: value},
    changes what needs changing, and gives it back.
    """
    firm  = row.get("Firm or Owner Name") or ""
    notes = row.get("Pay Day / Notes") or ""
    extra = row.get(EXTRA_COL) or ""

    if not firm:
        return row   # nothing to do if the field is empty

    # split on semicolons or newlines. both appear as delimiters in the original data
    parts         = re.split(r'[;\n]+', firm)
    keep_parts    = []
    contact_parts = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if looks_like_contact_info(part):
            contact_parts.append(part)   # flag it for moving
        else:
            keep_parts.append(part)      # actual firm name fragment, keep it

    if contact_parts:
        displaced = "; ".join(contact_parts)
        extra = send_to_extra(extra, displaced, "Firm or Owner Name")   # log it
        row["Firm or Owner Name"] = ", ".join(keep_parts) if keep_parts else None
        row["Pay Day / Notes"]    = append_note(notes, displaced)       # move it
        row[EXTRA_COL]            = extra

    return row


def fix_contact_in_firm_name(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Runs the contact-in-firm-name fix across every row."""
    target_cols = ["Firm or Owner Name", "Pay Day / Notes", EXTRA_COL]

    df = lf.collect()   # step 1; we need a real df for map_rows()

    updated = (
        df.select(target_cols)   # step 2; only the columns we're touching
          .map_rows(
              # step 3; convert each row tuple to a dict, run the fix,
              # then unpack the result back to a tuple in the same column order
              lambda row: tuple(
                  _fix_contact_firm_row(dict(zip(target_cols, row))).get(c)
                  for c in target_cols
              )
          )
          # step 4; map_rows() names columns column_0, column_1, etc.
          # this renames them back using their index position
          .rename({f"column_{i}": c for i, c in enumerate(target_cols)})
    )

    for col in target_cols:
        df = df.with_columns(updated[col].alias(col))   # step 5; write back

    return df.lazy()   # step 6; back to lazy so the pipeline can continue


# =============================================================================
# FIX 3 — WEFT / TWIST FIXES
# Same map_rows() approach as Fix 2. Checks the Weft and Twist columns
# against what the Details field actually says. It handles three situations:
#   3a. One value meant to apply to both columns (in the data this appears as
#       "xx/yy weft and twist" or "twist and weft xx/yy" in the Details field)
#   3b. Two separate values in Details — fix if they're swapped, fill if missing
#   3c. A number sitting in Weft with no mention of weft in Details is
#       inconsistent with the source field and is moved to the audit column
#
# On the data:
# Worrall's gives weft and twist counts inline within the prose entry, in
# forms like "15s/24s weft, 18s/20s twist" or "20s/40s weft and twist"
# (the latter meaning one count applies to both). Claude's fixed-template
# prompt extracted these into two separate structured columns, but the
# inline format made it error-prone: values are sometimes swapped between
# columns, sometimes only one is extracted when two were present, and
# occasionally a single combined value is placed in only one column.
# The Weft / Twist Details column preserves Claude's raw extraction of
# the inline text, which is used here as the truth to check and
# correct the structured columns against. Case 3c handles a seperate
# problem: a numeric value in the Weft column that Details never associates
# with weft is likely a column-shift imputation error. It is moved to the
# audit column rather than silently dropped or left in place.
# =============================================================================

def _fix_weft_twist_row(row: dict) -> dict:
    """Row-level weft/twist check and correction."""
    weft    = row.get("Weft (?)") or ""
    twist   = row.get("Twist") or ""
    details = row.get("Weft / Twist Details") or ""
    extra   = row.get(EXTRA_COL) or ""

    # 3a — look for a phrase like "xx/yy weft and twist" meaning one value for both.
    # the second pattern handles reversed word order ("twist and weft xx/yy"),
    # which appears regularly in combined-value entries in this dataset, i.e.,
    # it's a second structural case.
    combined_match = re.search(
        r'(\d[\d/\*]*)\s+(?:weft\s+and\s+twist|twist\s+and\s+weft)',
        details, re.IGNORECASE
    ) or re.search(
        r'(?:weft\s+and\s+twist|twist\s+and\s+weft)\s+([\d/\*]+)',
        details, re.IGNORECASE
    )

    if combined_match:
        shared = combined_match.group(1)
        row["Weft (?)"] = shared
        row["Twist"]    = shared
        return row   # done for this row, no need to keep checking

    # 3b — pull out whatever weft and twist values Details mentions separately
    if details:
        dw = re.search(r'weft[^\d]*(\d[\d/\*]*)',  details, re.IGNORECASE)
        dt = re.search(r'twist[^\d]*(\d[\d/\*]*)', details, re.IGNORECASE)

        if dw and dt:
            exp_weft, exp_twist = dw.group(1), dt.group(1)

            # if the two columns are just swapped, swap them back.
            # this is the most common structured-column error: Claude
            # extracted the right values from the Worrall's inline text
            # but placed them in the wrong fields.
            if weft == exp_twist and twist == exp_weft:
                row["Weft (?)"], row["Twist"] = twist, weft
                weft, twist = row["Weft (?)"], row["Twist"]

            # if weft is holding the twist value and twist is empty, fix both.
            # happens when Claude extracted only one value from the inline
            # text and defaulted to placing it in the first available column.
            if not twist and weft == exp_twist:
                row["Twist"], row["Weft (?)"] = exp_twist, exp_weft
                weft, twist = row["Weft (?)"], row["Twist"]

            # fill in whichever one is still missing
            if not row.get("Weft (?)"): row["Weft (?)"] = exp_weft
            if not row.get("Twist"):    row["Twist"]    = exp_twist

        # only one was found, just fill in the missing one
        elif dt and not twist: row["Twist"]    = dt.group(1)
        elif dw and not weft:  row["Weft (?)"] = dw.group(1)

    # 3c — if there's a number in Weft but Details never mentions weft,
    # the value is inconsistent with the source field. most likely a column-shift
    # error during transcription. move it to the audit column rather than leave
    # potentially misleading data in place — it stays recoverable there.
    weft = row.get("Weft (?)") or ""
    if weft and details and "weft" not in details.lower():
        if re.fullmatch(r'[\d/\*,\s]+', weft.strip()):
            row[EXTRA_COL]  = send_to_extra(extra, weft, "Weft (?)")
            row["Weft (?)"] = None   # clear the cell rather than leave bad data

    return row


def fix_weft_twist(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Runs the weft/twist correction across every row."""
    target_cols = ["Weft (?)", "Twist", "Weft / Twist Details", EXTRA_COL]

    df = lf.collect()

    updated = (
        df.select(target_cols)
          .map_rows(
              lambda row: tuple(
                  _fix_weft_twist_row(dict(zip(target_cols, row))).get(c)
                  for c in target_cols
              )
          )
          .rename({f"column_{i}": c for i, c in enumerate(target_cols)})
    )

    for col in target_cols:
        df = df.with_columns(updated[col].alias(col))

    return df.lazy()