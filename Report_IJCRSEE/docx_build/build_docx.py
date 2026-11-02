"""Build the IJCRSEE .docx from the Report_IJCRSEE LaTeX sources, on top of the real journal template.

Usage:  python docx_build/build_docx.py          (run from Report_IJCRSEE/)
Output: Report_IJCRSEE/SANF-Net_IJCRSEE.docx

The journal template (IJCRSEE_template.docx, converted from IJCRSEEtemplate.doc) supplies the page
setup, running header, footer, header box with "Check for updates", line numbering, the CC BY note and
every paragraph format. This script replaces the template's placeholder text and appends the report
body by cloning the template's own paragraphs (section heading, body text, table label/caption,
figure caption, reference entry), so the output follows the template exactly.
"""
import copy
import sys
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "docx_build"
TEMPLATE = BUILD / "IJCRSEE_template.docx"
OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "SANF-Net_IJCRSEE.docx")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
RED = "C00000"

TITLE = ("Dementia Detection Using State-Aware, Age-Normative EEG Classification with a Lightweight "
         "Fusion Network")
HEADER_CITE = ("Potdar, S., Raut, S., & Kannurkar, V. (", "202", "6",
               "). Dementia detection using state-aware, age-normative EEG classification with a "
               "lightweight fusion network. ")
AUTHORS = [("Siddarth Potdar", "1*"), ("Sanchit Raut", "1"), ("Vedant Kannurkar", "1")]
AFFIL = ("Department of Computer Science and Engineering, Sardar Patel Institute of Technology, Mumbai, "
         "India, e-mail: siddarth.potdar23@spit.ac.in, sanchit.raut23@spit.ac.in, vedant.kannurkar24@spit.ac.in")
CORRESPONDING = "siddarth.potdar23@spit.ac.in"

SECTIONS = ["01_introduction", "02_literature_survey", "03_materials_methods", "04_baselines",
            "05_proposed_model", "06_ensemble_results", "08_discussion",
            "09_conclusions", "10_declarations"]   # add "07_dashboard" back once the dashboard is built
RECEIVED = "Received: November 3, 2026."
BACKMATTER = {"Acknowledgements", "Funding", "Conflict of Interests", "Data Availability Statement",
              "Institutional Review Board Statement", "Author Contributions"}
FIG_IMG = {"figures/alignment.pdf": "alignment.png", "figures/protocol.pdf": "protocol.png",
           "figures/roc.pdf": "roc.png", "figures/confusion.pdf": "confusion.png"}
EQ_TEX = {1: r"\hat{t}_i = t_i - \sum_{p:\, r_p \leq t_i} (r_p - s_p)",
          2: r"z_j = \frac{x_j - f_j(\mathrm{age})}{1.4826\,\mathrm{MAD}_j}"}

for f in ("ARIALN.TTF", "ARIALNB.TTF"):
    fp = Path("C:/Windows/Fonts") / f
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))

# ============================================================================ images

def draw_architecture(path):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.stretch"] = "condensed"
    plt.rcParams["mathtext.fontset"] = "stix"
    fig, ax = plt.subplots(figsize=(9.0, 3.0), dpi=250)
    ax.set_xlim(-1.15, 9.9); ax.set_ylim(-2.7, 0.85); ax.axis("off")
    col = {"eeg": "#e3ecf8", "bio": "#fcecdc", "age": "#efefef", "fus": "#dcf1ee", "out": "#ffffff"}
    edge = {"eeg": "#4a72b0", "bio": "#c9772b", "age": "#7a7a7a", "fus": "#2f8f83", "out": "#333333"}

    def box(x, y, w, h, text, k, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08", fc=col[k], ec=edge[k], lw=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5, weight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.9, mutation_scale=9))

    ya, yb, yc = 0.0, -1.15, -2.1
    yf = (yb + yc) / 2
    box(0, ya, 1.95, 0.78, "EEG segments\n19 × 1000, state $s$", "eeg", True)
    box(0, yb, 1.95, 0.78, "48 age-normative\nbiomarkers", "bio", True)
    box(0, yc, 1.95, 0.78, "Age", "age", True)
    box(2.3, ya, 1.65, 0.78, "Multi-scale\nCNN + SE", "eeg")
    box(3.8, ya, 0.95, 0.78, "FiLM\n(state)", "eeg")
    box(5.15, ya, 1.25, 0.78, "Gated\nattention pool", "eeg")
    box(2.3, yb, 1.65, 0.78, "MLP → 64", "bio")
    box(2.3, yc, 1.65, 0.78, "MLP → 16", "age")
    box(5.15, yf, 1.45, 1.75, "Modality-\ngated fusion\n$\\sum_m g_m h_m$", "fus", True)
    box(7.05, yb, 1.5, 0.78, "Softmax head\n(focal loss)", "fus")
    box(7.05, yc, 1.5, 0.78, "Ordinal head\n$\\sigma(s - t_k)$", "fus")
    box(8.95, yf, 1.4, 0.95, "Average\nNormal / MCI /\nDementia", "out", True)
    arrow(0.98, ya, 1.47, ya); arrow(3.13, ya, 3.32, ya); arrow(4.28, ya, 4.52, ya)
    arrow(5.15, ya - 0.39, 5.15, yf + 0.88)
    arrow(0.98, yb, 1.47, yb); arrow(0.98, yc, 1.47, yc)
    arrow(3.13, yb, 4.42, yb); arrow(3.13, yc, 4.42, yc)
    arrow(5.88, yb, 6.3, yb); arrow(5.88, yc, 6.3, yc)
    arrow(7.8, yb, 8.25, yf + 0.2); arrow(7.8, yc, 8.25, yf - 0.2)
    ax.text(2.3, ya + 0.55, "shared encoder, 12–24 segments per recording", ha="center", fontsize=8,
            color="#666666", style="italic")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_equation(tex, path):
    plt.rcParams["mathtext.fontset"] = "stix"
    fig = plt.figure(figsize=(0.01, 0.01), dpi=400)
    fig.text(0, 0, f"${tex}$", fontsize=14)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)

# ============================================================================ bibliography (APA)

def parse_bib(path):
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", path.read_text(encoding="utf-8"), re.S):
        fields = {"_type": m.group(1).lower()}
        for fm in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", m.group(3)):
            fields[fm.group(1).lower()] = fm.group(2)
        entries[m.group(2).strip()] = fields
    return entries


def tex_clean(s):
    for a, b in ((r"{\c{c}}", "ç"), (r"\c{c}", "ç"), (r"{\"o}", "ö"), (r"\"o", "ö"), (r"{\"e}", "ë"),
                 (r"\"e", "ë"), (r"{\'a}", "á"), (r"\'a", "á"), ("--", "–"), (r"\url", ""), (r"\&", "&")):
        s = s.replace(a, b)
    return re.sub(r"[{}]", "", s).strip()


def split_authors(a):
    return [x.strip() for x in tex_clean(a).split(" and ")]


def surname(a):
    return a.split(",")[0].strip() if "," in a else (a.split()[-1] if a.split() else a)


def cite_names(entry):
    if entry.get("author", "").startswith("{"):
        return tex_clean(entry["author"])
    auths = split_authors(entry.get("author", "Anon."))
    if auths[-1].lower() == "others" or len(auths) >= 3:
        return f"{surname(auths[0])} et al."
    if len(auths) == 2:
        return f"{surname(auths[0])} & {surname(auths[1])}"
    return surname(auths[0])


def apa_author(a):
    if a.lower() == "others":
        return None
    if a.startswith("TODO") or "World Health" in a:
        return a
    last, first = ([x.strip() for x in a.split(",", 1)] if "," in a else (a.split()[-1], " ".join(a.split()[:-1])))
    initials = " ".join("-".join(q[0].upper() + "." for q in p.split("-") if q)
                        for p in first.split() if p and p[0].isalpha())
    return f"{last}, {initials}".strip().rstrip(",")


def apa_reference(entry):
    auths = [apa_author(a) for a in split_authors(entry.get("author", "Anon."))]
    etal = None in auths
    auths = [a for a in auths if a]
    if len(auths) == 1:
        astr = auths[0] + (", et al." if etal else "")
    elif etal:
        astr = ", ".join(auths) + ", et al."
    else:
        astr = ", ".join(auths[:-1]) + ", & " + auths[-1]
    title = tex_clean(entry.get("title", "")).rstrip(".")
    title_end = "" if title.endswith(("?", "!")) else "."
    runs = [(f"{astr} ({entry.get('year', 'n.d.')}). ", {})]
    typ = entry["_type"]
    if typ == "article":
        runs += [(f"{title}{title_end} ", {}), (tex_clean(entry.get("journal", "")), {"i": True})]
        if entry.get("volume"):
            runs += [(", ", {}), (entry["volume"], {"i": True})]
        if entry.get("number"):
            runs.append((f"({entry['number']})", {}))
        if entry.get("pages"):
            runs.append((f", {tex_clean(entry['pages'])}", {}))
        runs.append((".", {}))
    elif typ == "inproceedings":
        runs += [(f"{title}{title_end} In ", {}), (tex_clean(entry.get("booktitle", "")), {"i": True})]
        if entry.get("pages"):
            runs.append((f" (pp. {tex_clean(entry['pages'])})", {}))
        runs.append((".", {}))
    else:
        how = tex_clean(entry.get("howpublished", ""))
        runs += [(title, {"i": True}), (f". {how}." if how else ".", {})]
    return [(t, dict(f, color=RED) if "TODO" in t else f) for t, f in runs]

# ============================================================================ LaTeX inline -> runs
# A run is (text, fmt) with fmt keys: b, i, color, va ("subscript"/"superscript").

SYM = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "theta": "θ", "sigma": "σ", "mu": "μ",
       "pm": "±", "times": "×", "odot": "⊙", "le": "≤", "leq": "≤", "ge": "≥", "downarrow": "↓",
       "rightarrow": "→", "to": "→", "sum": "Σ", "log": "log", "ast": "*", "dagger": "†", "Delta": "Δ", "cdot": "·"}


def read_group(s, i):
    while i < len(s) and s[i] == " ":
        i += 1
    if i < len(s) and s[i] == "\\":           # single command token, e.g. _\alpha
        m = re.match(r"\\[a-zA-Z]+|\\.", s[i:])
        return m.group(0), i + m.end()
    if i >= len(s) or s[i] != "{":
        return (s[i] if i < len(s) else ""), i + 1
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        elif s[j] == "\\":
            j += 1
        j += 1
    return s[i + 1:], len(s)


def math_runs(m, fmt):
    """Inline math -> runs with real sub/superscripts."""
    out, buf = [], []

    def flush(va=None, text=None):
        if buf:
            t = "".join(buf)
            if t.strip() not in ("<", ">", "=", "≤"):          # lone operator: text already has spaces
                t = re.sub(r"\s*([=<>≤])\s*", r" \1 ", t).replace("  ", " ")
            t = re.sub(r"\s*±\s*", "±", t)                     # 68.4±2.8, as in the LaTeX output
            out.append((t, dict(fmt)))
            buf.clear()
        if text is not None:
            out.append((text, dict(fmt, va=va)))

    i = 0
    while i < len(m):
        c = m[i]
        if c == "\\":
            mm = re.match(r"\\([a-zA-Z]+)|\\(.)", m[i:], re.S)
            name = mm.group(1) or mm.group(2)
            i += mm.end()
            if name in ("mathrm", "text"):
                g, i = read_group(m, i); buf.append(g)
            elif name == "hat":
                g, i = read_group(m, i); buf.append(g + "\u0302")
            elif name in SYM:
                buf.append(SYM[name])
            elif name in (",", " "):
                buf.append("\u2009")
            elif name == "%":
                buf.append("%")
            continue
        if c in "_^":
            g, i = read_group(m, i + 1)
            g = "".join(t for t, _ in math_runs(g, {}))
            flush("subscript" if c == "_" else "superscript", g)
            continue
        if c in "{}":
            i += 1; continue
        buf.append(c); i += 1
    flush()
    return out


class Ctx:
    def __init__(self, bib, labels):
        self.bib, self.labels = bib, labels


def fmt_cite(ctx, keys, mode):
    parts = [(cite_names(ctx.bib[k]), ctx.bib[k].get("year", "n.d.")) if k in ctx.bib else (k, "?")
             for k in (x.strip() for x in keys.split(","))]
    if mode == "t":
        return "; ".join(f"{n} ({y})" for n, y in parts)
    return "(" + "; ".join(f"{n}, {y}" for n, y in parts) + ")"


ESC = {"%": "%", "&": "&", "_": "_", ",": "\u2009", " ": " ", "\n": " ", "\\": "", "#": "#"}


def inline(s, ctx, fmt=None):
    fmt = dict(fmt or {})
    runs, buf, i = [], [], 0

    def flush():
        if buf:
            runs.append(("".join(buf), dict(fmt))); buf.clear()

    while i < len(s):
        c = s[i]
        if c == "$":
            j = s.index("$", i + 1)
            flush(); runs.extend(math_runs(s[i + 1:j], fmt)); i = j + 1; continue
        if c == "~":
            buf.append("\u00a0"); i += 1; continue
        if s.startswith("---", i):
            buf.append("—"); i += 3; continue
        if s.startswith("--", i):
            buf.append("–"); i += 2; continue
        if s.startswith("``", i) or s.startswith("''", i):
            buf.append("\""); i += 2; continue
        if c in "{}":
            i += 1; continue
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+)\*?|\\(.)", s[i:], re.S)
            if m is None:
                i += 1; continue
            name = m.group(1) or m.group(2)
            i += m.end()
            if m.group(2):
                buf.append(ESC.get(name, name)); continue
            if name in ("footnotesize", "small", "scriptsize", "noindent", "centering"):
                continue
            if name in ("textbf", "textit", "emph", "todo", "textcolor", "url", "texttt", "citep", "citet",
                        "ref", "label"):
                if name == "textcolor":
                    _, i = read_group(s, i)
                arg, i = read_group(s, i)
                flush()
                if name == "textbf":
                    runs.extend(inline(arg, ctx, dict(fmt, b=True)))
                elif name in ("textit", "emph"):
                    runs.extend(inline(arg, ctx, dict(fmt, i=True)))
                elif name in ("todo", "textcolor"):
                    runs.extend(inline(("[TODO: " + arg + "]") if name == "todo" else arg, ctx, dict(fmt, color=RED)))
                elif name in ("url", "texttt"):
                    runs.append((arg, dict(fmt)))
                elif name in ("citep", "citet"):
                    runs.append((fmt_cite(ctx, arg, name[-1]), dict(fmt)))
                elif name == "ref":
                    runs.append((str(ctx.labels.get(arg, "?")), dict(fmt)))
                continue
            if name in SYM:
                buf.append(SYM[name])
            continue
        buf.append(c); i += 1
    flush()
    out = []
    for t, f in runs:
        t = re.sub(r"[ \t\n]+", " ", t)
        if t:
            out.append((t, f))
    if out:
        out[0] = (out[0][0].lstrip(), out[0][1])
        out[-1] = (out[-1][0].rstrip(), out[-1][1])
    return [r for r in out if r[0]]

# ============================================================================ template cloning

class Builder:
    def __init__(self, doc):
        self.doc = doc
        self.body = doc.element.body
        b = list(self.body)
        texts = [self._text(el) for el in b]

        def find(prefix, start=0):
            for k in range(start, len(b)):
                if texts[k].startswith(prefix):
                    return k
            raise KeyError(prefix)

        self.idx = {"intro": find("Introduction"), "ack": find("Acknowledgements"),
                    "tlabel": find("Table 1"), "figcap": find("Figure 1."), "ref": find("McCormick")}
        self.idx["body"] = self.idx["intro"] + 3        # "Introduction is the first section..."
        self.idx["blank"] = self.idx["intro"] + 1
        self.idx["tcap"] = self.idx["tlabel"] + 1
        self.proto = {k: copy.deepcopy(b[v]) for k, v in self.idx.items()}
        self.proto_table = copy.deepcopy(b[self.idx["tcap"] + 1])
        self.sect = self.body[-1]
        self.b = b

    @staticmethod
    def _text(el):
        return "".join(t.text or "" for t in el.iter(W + "t"))

    # -- element factories ---------------------------------------------------
    def _new_from(self, key):
        p = copy.deepcopy(self.proto[key])
        base_rpr = None
        for r in p.findall(W + "r"):
            if base_rpr is None and r.find(W + "rPr") is not None:
                base_rpr = copy.deepcopy(r.find(W + "rPr"))
        for child in list(p):
            if child.tag != W + "pPr":
                p.remove(child)
        if base_rpr is None:
            ppr_rpr = p.find(W + "pPr/" + W + "rPr")
            base_rpr = copy.deepcopy(ppr_rpr) if ppr_rpr is not None else OxmlElement("w:rPr")
        for tag in ("b", "bCs", "i", "iCs", "color", "vertAlign", "rStyle"):
            for el in base_rpr.findall(W + tag):
                base_rpr.remove(el)
        return p, base_rpr

    @staticmethod
    def _run(base_rpr, text, f, size=None):
        r = OxmlElement("w:r")
        rpr = copy.deepcopy(base_rpr)
        def add(tag, val=None):
            el = OxmlElement(f"w:{tag}")
            if val is not None:
                el.set(qn("w:val"), val)
            rpr.append(el)
        if f.get("b"):
            add("b"); add("bCs")
        if f.get("i"):
            add("i"); add("iCs")
        if f.get("color"):
            add("color", f["color"])
        if size:
            for tag in ("sz", "szCs"):
                for el in rpr.findall(W + tag):
                    rpr.remove(el)
                add("sz", str(size * 2)); add("szCs", str(size * 2))
        if f.get("va"):
            add("vertAlign", f["va"])
        # schema order inside rPr: rStyle, rFonts, b, bCs, i, iCs, ..., color, ..., sz, szCs, ..., vertAlign
        order = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "color", "spacing", "kern",
                 "sz", "szCs", "u", "vertAlign", "lang"]
        kids = sorted(list(rpr), key=lambda e: order.index(e.tag.replace(W, "")) if e.tag.replace(W, "") in order else 99)
        for k in list(rpr):
            rpr.remove(k)
        for k in kids:
            rpr.append(k)
        r.append(rpr)
        t = OxmlElement("w:t")
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text
        r.append(t)
        return r

    def _append(self, el):
        self.sect.addprevious(el)
        return el

    def para(self, key, runs, size=None, after_setup=None):
        p, base = self._new_from(key)
        for text, f in runs:
            p.append(self._run(base, text, f, size))
        if after_setup:
            after_setup(p)
        return self._append(p)

    def blank(self):
        p, _ = self._new_from("blank")
        return self._append(p)

    @staticmethod
    def _keep_next(p):
        ppr = p.find(W + "pPr")
        if ppr.find(W + "keepNext") is None:
            ppr.insert(1 if ppr.find(W + "pStyle") is not None else 0, OxmlElement("w:keepNext"))

    def heading1(self, text):
        self._keep_next(self.blank())
        self.para("intro", [(text, {"b": True})], after_setup=self._keep_next)
        self._keep_next(self.blank())

    def heading2(self, text, italic=True):
        def keep(p):
            self._keep_next(p)
            sp = p.find(f"{W}pPr/{W}spacing")
            sp.set(qn("w:before"), "160"); sp.set(qn("w:after"), "40")
        self.para("ack", [(text, {"b": True, "i": italic})], after_setup=keep)

    def body_par(self, runs, indent=True, hanging=None):
        def setup(p):
            ind = p.find(W + "pPr/" + W + "ind")
            if ind is None:
                return
            if not indent:
                ind.set(qn("w:firstLine"), "0")
            if hanging:
                for a in ("firstLine",):
                    if ind.get(qn(f"w:{a}")) is not None:
                        del ind.attrib[qn(f"w:{a}")]
                ind.set(qn("w:left"), str(hanging[0])); ind.set(qn("w:hanging"), str(hanging[1]))
        return self.para("body", runs, after_setup=setup)

    def picture(self, path, width_cm):
        p = self.doc.add_paragraph()
        p.alignment = 1
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        self.sect.addprevious(p._p)          # (add_paragraph already put it before sectPr)
        return p

    def figure(self, ctx, img, width_frac, caption, label):
        self.picture(BUILD / img, 16.0 * width_frac)
        num = ctx.labels[label]
        self.para("figcap", [(f"Figure {num}.", {"i": True}), (" ", {})] + inline(caption, ctx))
        self.blank()

    def table(self, ctx, caption, label, body, notes, colwidths):
        self.blank()
        def keep(p):
            ppr = p.find(W + "pPr"); ppr.insert(0, OxmlElement("w:keepNext"))
        self.para("tlabel", [(f"Table {ctx.labels[label]}", {"b": True})], after_setup=keep)
        self.para("tcap", inline(caption, ctx, {"i": True}), after_setup=keep)
        rows, rules = parse_tabular(body)
        ncol = max(sum(sp for _, sp in r) for r in rows)
        header_rows = min(rules) + 1 if rules else 1
        font = 9 if (label == "tab:lit" or ncol >= 7) else 10
        if colwidths == "auto":     # size each column to its content (compact, like the template's tables)
            longest = [0.0] * ncol
            for r in rows:
                ci = 0
                for text, span in r:
                    if span == 1:
                        n = len("".join(t for t, _ in inline(text, ctx)))
                        longest[ci] = max(longest[ci], min(n, 34))
                    ci += span
            cm_per_char = 0.165 if font == 10 else 0.15
            colwidths = [max(1.5, n * cm_per_char + 0.5) for n in longest]
            if sum(colwidths) > 16.0:
                colwidths = [w * 16.0 / sum(colwidths) for w in colwidths]
        tbl = OxmlElement("w:tbl")
        tblPr = copy.deepcopy(self.proto_table.find(W + "tblPr"))
        tblPr.find(W + "tblW").set(qn("w:w"), str(int(sum(colwidths) * 567)))
        tbl.append(tblPr)
        grid = OxmlElement("w:tblGrid")
        for w in colwidths:
            gc = OxmlElement("w:gridCol"); gc.set(qn("w:w"), str(int(w * 567))); grid.append(gc)
        tbl.append(grid)
        cell_p_proto = self.proto_table.find(f"{W}tr/{W}tc/{W}p")
        for ri, r in enumerate(rows):
            tr = OxmlElement("w:tr")
            trPr = OxmlElement("w:trPr"); trPr.append(OxmlElement("w:cantSplit"))
            if ri < header_rows:              # header repeats if the table breaks across pages
                trPr.append(OxmlElement("w:tblHeader"))
            tr.append(trPr)
            ci = 0
            for text, span in r:
                tc = OxmlElement("w:tc")
                tcPr = OxmlElement("w:tcPr")
                tcW = OxmlElement("w:tcW"); tcW.set(qn("w:w"), str(int(sum(colwidths[ci:ci + span]) * 567)))
                tcW.set(qn("w:type"), "dxa"); tcPr.append(tcW)
                if span > 1:
                    gs = OxmlElement("w:gridSpan"); gs.set(qn("w:val"), str(span)); tcPr.append(gs)
                if ri in rules and ri != len(rows) - 1:
                    tb = OxmlElement("w:tcBorders"); bt = OxmlElement("w:bottom")
                    for a, v in (("val", "single"), ("sz", "4"), ("space", "0"), ("color", "auto")):
                        bt.set(qn(f"w:{a}"), v)
                    tb.append(bt); tcPr.append(tb)
                va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center"); tcPr.append(va)
                tc.append(tcPr)
                p = copy.deepcopy(cell_p_proto)
                for child in list(p):
                    if child.tag != W + "pPr":
                        p.remove(child)
                if ri < min(header_rows + 2, len(rows) - 1) or len(rows) <= 6:   # header + first rows stay together
                    p.find(W + "pPr").insert(0, OxmlElement("w:keepNext"))
                left = (ci == 0 and span == 1) or label == "tab:lit"
                p.find(f"{W}pPr/{W}jc").set(qn("w:val"), "left" if left else "center")
                base = copy.deepcopy(p.find(f"{W}pPr/{W}rPr"))
                for text_, f in inline(text, ctx):
                    p.append(self._run(base, text_, f, font))
                tc.append(p)
                tr.append(tc)
                ci += span
            tbl.append(tr)
        self._append(tbl)
        if notes:
            self.para("figcap", inline(notes, ctx), size=9)
        self.blank()

    def remove_range(self, start, stop, keep=()):
        for el in self.b[start:stop]:
            if el not in keep and el is not self.sect:
                self.body.remove(el)

# ============================================================================ tables / sections

def split_row(row):
    cells, depth, cur, i = [], 0, [], 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row):
            cur.append(row[i:i + 2]); i += 2; continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            cells.append("".join(cur)); cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur))
    return cells


def parse_tabular(body):
    body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
    rows, rules = [], set()
    for raw in re.split(r"\\\\(?:\[[^\]]*\])?", body):
        r = raw.strip()
        for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
            if rule in r:
                r = r.replace(rule, "").strip()
                if rows:
                    rules.add(len(rows) - 1)
        if not r:
            continue
        cells = []
        for c in split_row(r):
            c = c.strip()
            if c.startswith("\\multicolumn"):
                n, i = read_group(c, len("\\multicolumn"))
                _, i = read_group(c, i)
                content, _ = read_group(c, i)
                cells.append((content, int(n)))
            else:
                cells.append((c, 1))
        rows.append(cells)
    return rows, rules


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def collect_labels(texts):
    labels, nt, nf = {}, 0, 0
    for t in texts:
        for m in re.finditer(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", t, re.S):
            lab = re.search(r"\\label\{([^}]*)\}", m.group(0)).group(1)
            if m.group(1) == "table":
                nt += 1; labels[lab] = nt
            else:
                nf += 1; labels[lab] = nf
    return labels


def emit_paragraphs(B, chunk, ctx):
    for block in re.split(r"\n\s*\n", chunk):
        if block.strip():
            runs = inline(block.strip(), ctx)
            if runs:
                B.body_par(runs)


def convert(B, text, ctx, state):
    text = strip_comments(text)
    pos = 0
    pattern = re.compile(r"\\section\{(.*?)\}|\\subsection\{(.*?)\}|"
                         r"\\begin\{(table|figure|equation|itemize|enumerate)\}(\[[^\]]*\])?(.*?)\\end\{\3\}", re.S)
    for m in pattern.finditer(text):
        emit_paragraphs(B, text[pos:m.start()], ctx)
        pos = m.end()
        if m.group(1) is not None:
            title = "".join(t for t, _ in inline(m.group(1), ctx))
            if title in BACKMATTER:
                B.blank(); B.heading2(title, italic=False)
            else:
                B.heading1(title)
        elif m.group(2) is not None:
            B.heading2("".join(t for t, _ in inline(m.group(2), ctx)))
        else:
            env, body = m.group(3), m.group(5)
            if env == "equation":
                state["eq"] += 1
                img = BUILD / f"eq{state['eq']}.png"
                draw_equation(EQ_TEX[state["eq"]], img)
                # equation centred on the text block, number flush right (centre + right tab stops)
                p = B.doc.add_paragraph()
                pf = p.paragraph_format
                pf.space_before, pf.space_after = Cm(0.15), Cm(0.15)
                pf.tab_stops.add_tab_stop(Cm(8.0), alignment=1)
                pf.tab_stops.add_tab_stop(Cm(15.9), alignment=2)
                p.add_run("\t")
                p.add_run().add_picture(str(img), height=Cm(1.35 if state["eq"] == 2 else 1.05))
                num = p.add_run(f"\t({state['eq']})")
                num.font.name = "Arial Narrow"; num.font.size = Cm(0.42)
                B.sect.addprevious(p._p)
            elif env in ("itemize", "enumerate"):
                for k, it in enumerate([x.strip() for x in re.split(r"\\item", body)[1:]], 1):
                    mark = f"{k}." if env == "enumerate" else "–"
                    B.body_par([(mark + "\t", {})] + inline(it, ctx), hanging=(851, 284))
            elif env == "table":
                cap = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", body, re.S).group(1)
                lab = re.search(r"\\label\{([^}]*)\}", body).group(1)
                t0 = re.search(r"\\begin\{(tabularx?)\}", body)
                i = t0.end()
                for _ in range(2 if t0.group(1) == "tabularx" else 1):
                    _, i = read_group(body, i)
                t1 = body.index(f"\\end{{{t0.group(1)}}}", i)
                notes = re.search(r"\\parbox\{[^}]*\}\{\\scriptsize\s*(.*)\}\s*$", body.strip(), re.S)
                widths = [2.4, 2.3, 4.6, 2.3, 4.4] if lab == "tab:lit" else "auto"
                B.table(ctx, cap, lab, body[i:t1], notes.group(1) if notes else None, widths)
            elif env == "figure":
                cap = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", body, re.S).group(1)
                lab = re.search(r"\\label\{([^}]*)\}", body).group(1)
                g = re.search(r"\\includegraphics\[width=([0-9.]+)\\textwidth\]\{([^}]*)\}", body)
                if g:
                    B.figure(ctx, FIG_IMG[g.group(2)], float(g.group(1)), cap, lab)
                elif "architecture" in body:
                    B.figure(ctx, "architecture.png", 1.0, cap, lab)
                else:   # dashboard placeholder
                    B.para("figcap", [("[Dashboard screenshot placeholder]", {"color": RED})])
                    B.para("figcap", [(f"Figure {ctx.labels[lab]}.", {"i": True}), (" ", {})] + inline(cap, ctx))
                    B.blank()
    emit_paragraphs(B, text[pos:], ctx)

# ============================================================================ front matter

def set_runs(B, p, runs, size=None):
    """Replace the text of an existing template paragraph, keeping its first run's format."""
    base = None
    for r in p.findall(W + "r"):
        if r.find(W + "rPr") is not None:
            base = copy.deepcopy(r.find(W + "rPr")); break
    for child in list(p):
        if child.tag != W + "pPr":
            p.remove(child)
    if base is None:
        base = OxmlElement("w:rPr")
    for tag in ("b", "bCs", "i", "iCs", "vertAlign"):
        for el in base.findall(W + tag):
            base.remove(el)
    for text, f in runs:
        p.append(B._run(base, text, f, size))


def main():
    draw_architecture(BUILD / "architecture.png")
    bib = parse_bib(ROOT / "references.bib")
    texts = {s: (ROOT / "sections" / f"{s}.tex").read_text(encoding="utf-8") for s in SECTIONS}
    ctx = Ctx(bib, collect_labels(texts.values()))

    doc = Document(str(TEMPLATE))
    B = Builder(doc)
    b = B.b

    # running header
    hp = doc.sections[0].header.paragraphs[0]
    for r, t in zip(hp.runs[:4], HEADER_CITE):
        r.text = t

    def at(prefix):
        for el in b:
            if B._text(el).startswith(prefix):
                return el
        raise KeyError(prefix)

    # header box: real submission date; drop the journal-assigned placeholders (revision/acceptance dates, UDC, doi)
    box = b[0] if b[0].tag == W + "tbl" else next(el for el in b if el.tag == W + "tbl")
    for p in list(box.iter(W + "p")):
        t = B._text(p).strip()
        if t.startswith("Received"):
            set_runs(B, p, [(RECEIVED, {})])
        elif t.startswith(("Revised", "Accepted", "UDC", "doi")) or t in ("xxx", ""):
            # also drops the empty spacer line and the journal's "Check for updates" badge (unpublished paper)
            cell = p.getparent()
            if len(cell.findall(W + "p")) > 1:
                cell.remove(p)
            else:
                for child in list(p):
                    if child.tag != W + "pPr":
                        p.remove(child)

    # title, authors, affiliation
    title_p = at("Paper Title")
    set_runs(B, title_p, [(TITLE, {"b": True})])
    sp = title_p.find(f"{W}pPr/{W}spacing")
    if sp is None:
        sp = OxmlElement("w:spacing"); title_p.find(W + "pPr").append(sp)
    sp.set(qn("w:line"), "264"); sp.set(qn("w:lineRule"), "auto")
    snap = OxmlElement("w:snapToGrid"); snap.set(qn("w:val"), "0")
    sp.addprevious(snap)                     # don't stretch the 16 pt title onto the 12 pt line grid
    nohyph = OxmlElement("w:suppressAutoHyphens")
    snap.addprevious(nohyph)                 # wrap the title on whole words
    auth = []
    for k, (name, sup) in enumerate(AUTHORS):
        auth += [(name, {}), (sup, {"va": "superscript"})] + ([(", ", {})] if k < len(AUTHORS) - 1 else [])
    set_runs(B, at("Author1"), auth)
    set_runs(B, at("1 Author Affiliation"), [("1", {"va": "superscript"}), (" " + AFFIL, {})])
    B.body.remove(at("2 Co-Author Affiliation"))

    # abstract + keywords
    abs_tex = strip_comments((ROOT / "sections" / "00_abstract.tex").read_text(encoding="utf-8"))
    a = re.search(r"\\textbf\{Abstract:\}(.*?)\\vspace", abs_tex, re.S).group(1)
    k = re.search(r"\\textbf\{Keywords:\}(.*)", abs_tex, re.S).group(1)
    abs_p = at("Abstract:")
    set_runs(B, abs_p, [("Abstract:", {"b": True}), (" ", {})] + inline(a, ctx), size=10)
    for lab in ("Purpose", "Methodology", "Results", "Conclusions", "Recommendations", "Additional data",
                "/Arial Narrow, 10/"):
        for el in b[b.index(abs_p) + 1:b.index(abs_p) + 9]:
            if B._text(el).strip() == lab and el.getparent() is not None:
                B.body.remove(el); break
    kw_p = at("Keywords:")
    set_runs(B, kw_p, [("Keywords:", {"b": True, "i": True}), (" ", {})] + inline(k, ctx))

    # corresponding author + CC licence note: keep the template's paragraphs, move them up
    corr = at("*Corresponding author")
    cc = corr.getnext()                       # licence paragraph with the CC BY logo
    sep = corr.getprevious()
    corr_runs = corr.findall(W + "r")
    # template: run 0 = superscript '*', following runs 'Corresponding author: '; append the e-mail
    last = copy.deepcopy(corr_runs[-1])
    for t in last.findall(W + "t"):
        t.text = CORRESPONDING
    corr.append(last)
    for rpr in corr.iter(W + "rPr"):          # template style here is not Arial Narrow; match the body text
        f = rpr.find(W + "rFonts")
        if f is None:
            f = OxmlElement("w:rFonts"); rpr.insert(0, f)
        for k in ("ascii", "hAnsi", "cs"):
            f.set(qn(f"w:{k}"), "Arial Narrow")

    # drop all template body content after the keywords (keeping the corresponding-author block)
    start = B.idx["intro"]
    keep = {sep, corr, cc}
    B.remove_range(start, len(b), keep=keep)
    # place corresponding-author block right after keywords, then a blank line
    anchor = kw_p
    for el in (sep, corr, cc):
        anchor.addnext(el); anchor = el

    # body
    state = {"eq": 0}
    for s in SECTIONS:
        convert(B, texts[s], ctx, state)

    # references
    B.blank()
    B.heading2("References", italic=False)
    B.blank()
    cited = set()
    for t in list(texts.values()) + [abs_tex]:
        for m in re.finditer(r"\\cite[pt]?\{([^}]*)\}", t):
            cited.update(x.strip() for x in m.group(1).split(","))
    keys = sorted((x for x in cited if x in bib),
                  key=lambda x: (tex_clean(bib[x].get("author", "")).lower(), bib[x].get("year", "")))
    for key in keys:
        B.para("ref", apa_reference(bib[key]), size=10)

    # final copy: no review line numbers in the margin
    for ln in B.sect.findall(W + "lnNumType"):
        B.sect.remove(ln)
    doc.save(OUT)
    for f in BUILD.glob("eq*.png"):
        f.unlink()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
