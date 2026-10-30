"""Theme + browser-side effects injection for Streamlit.

``st.markdown`` never executes ``<script>``. The supported route for
browser-side behaviour is a tiny component iframe whose script appends a
<script> element to the *parent* document (the iframe is same-origin), so
the code lives in the page's own JS realm and survives Streamlit reruns.
If anything here fails the app still works - it only loses the polish.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
_FONTS = ("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
          "&family=JetBrains+Mono:wght@400;500;600&display=swap');")


def inject_theme() -> None:
    css = (ASSETS / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{_FONTS}\n{css}</style>", unsafe_allow_html=True)


def inject_fx(enabled: bool = True) -> None:
    js = (ASSETS / "fx.js").read_text(encoding="utf-8")
    version = int(re.search(r"VERSION\s*=\s*(\d+)", js).group(1))
    html = f"""<script>
(function () {{
  try {{
    var p = window.parent, d = p.document;
    var code = {json.dumps(js)};
    if (!p.__ciFx || p.__ciFx.v !== {version}) {{
      var old = d.getElementById('ci-fx-script'); if (old) old.remove();
      var s = d.createElement('script'); s.id = 'ci-fx-script'; s.textContent = code;
      d.head.appendChild(s);
    }}
    if (p.__ciFx) {{ p.__ciFx.setEnabled({str(bool(enabled)).lower()}); p.__ciFx.refresh(); }}
  }} catch (err) {{ console.warn('[ci-fx] disabled:', err); }}
}})();
</script>"""
    try:
        from streamlit.components.v1 import html as _html
        with st.container(key="fx_host"):
            _html(html, height=0)
    except Exception:
        pass  # purely cosmetic
