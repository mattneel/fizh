#!/usr/bin/env python3
"""nonbreaking.py — non-breaking prefixes, per language. ADR 0011.

A period after one of these does not end a sentence. Derived from Moses'
`nonbreaking_prefix.*` lists, which is what bergamot-translator's ssplit-cpp
uses, trimmed to the entries that actually occur in running text.

Language knowledge belongs in the converter, not the runtime: `tok/ssplit.zig`
takes whatever list the artifact carries and has no opinion about which
language it is looking at.
"""

from __future__ import annotations

# Titles, honorifics and abbreviations shared across Latin-script languages.
_COMMON = """
mr mrs ms dr prof st jr sr rev hon gen col maj capt lt sgt gov pres
vs etc inc ltd co corp dept est fig no nos vol op cf al
jan feb mar apr jun jul aug sep sept oct nov dec
mon tue wed thu fri sat sun
i ii iii iv v vi vii viii ix x xi xii
""".split()

_BY_LANG = {
    "en": """
    adm approx apt asst attys ave bldg blvd brig bros capt cmdr comdr
    corp cpl det dist dr drs ed eng ens gen gov hon hosp insp lt mm mr mrs ms
    maj messrs mlle mme mr mrs msgr mssrs mt mts op ord pfc ph phd pvt rep
    reps res rev rt sen sens sfc sgt sr st supt surg univ
    """,
    "es": """
    apdo appx aprox av avda avda bco bibl brig cap cia cta dcha
    depto dna dpto dr dra dras dres ee ej esq etc excmo fig gob gral hnos
    ing izq izqda lic ltd ltda min núm num pág págs pdte ppal prof profa
    pza rda ref rte sr sra sras sres srta sta sto tel telf ud uds univ
    vda vol
    """,
    "de": """
    abb abk abs abt ahd allg alt anh anm art aufl bd bearb beil bes bez
    bspw bzgl bzw ca dgl dh dr ebd eigtl entspr erg evtl exkl geb gem ggf
    ggfs hg hrsg inkl insb jh jhd kap lfd lt max min mind mio mrd nr
    näml o.ä od op pfl pl prof rd resp s.o s.u sog st std str tsd u.a u.ä
    urspr usw uvm v.a vgl vs z.b z.t zb ztr zz
    """,
}


def prefixes(lang: str) -> list[str]:
    """Sorted, deduplicated, lowercase."""
    words = set(_COMMON)
    words.update(_BY_LANG.get(lang, "").split())
    # A prefix containing a period ("z.b") is matched on its last segment by
    # tok/ssplit.zig's word scan, so store that segment too.
    for w in list(words):
        if "." in w:
            words.add(w.rsplit(".", 1)[-1])
    return sorted(w for w in words if w and w.isprintable())


def blob(lang: str) -> bytes:
    """NUL-separated, as `tok.nonbreaking` ships it."""
    return b"".join(w.encode("utf-8") + b"\0" for w in prefixes(lang))


if __name__ == "__main__":
    import sys

    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    b = blob(lang)
    print(f"{lang}: {len(prefixes(lang))} prefixes, {len(b)} bytes")
    print("  " + " ".join(prefixes(lang)[:24]) + " ...")
