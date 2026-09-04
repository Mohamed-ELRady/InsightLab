"""Language-model access configured from the Streamlit sidebar.

Keys typed here live only in Streamlit session state. They are deliberately not
copied into ``os.environ`` or written to ``.env``: a hosted app may serve more
than one person, and a convenience feature must not turn one person's key into
another person's credential.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from ..core.config import PROVIDERS, Settings, get_settings, with_model_access
from ..core.language import Language
from ..core.reasoning import ReasoningEngine

CUSTOM_MODEL = "__custom_model__"


def _connection_fingerprint(settings: Settings) -> str:
    """Identify one exact connection setup without retaining another key copy."""
    parts = (
        settings.llm_provider,
        settings.llm_model,
        settings.llm_base_url or "",
        settings.api_key or "",
    )
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _copy(language: Language, english: str, arabic: str) -> str:
    return arabic if language.code == "ar" else english


def _provider_label(key: str, language: Language) -> str:
    spec = PROVIDERS[key]
    if not spec.free_tier:
        return spec.label
    suffix = "خطة مجانية" if language.code == "ar" else "free tier"
    return f"{spec.label} · {suffix}"


def _initialise_provider(base: Settings) -> str:
    provider = st.session_state.get("llm_provider", base.llm_provider)
    if provider not in PROVIDERS:
        provider = base.llm_provider
    st.session_state.setdefault("llm_provider", provider)
    return provider


def _model_for(provider: str, base: Settings) -> str:
    """Resolve the selected suggestion or the user's custom model ID."""
    spec = PROVIDERS[provider]
    custom_key = f"llm_custom_model_{provider}"

    if not spec.models:
        if custom_key not in st.session_state:
            initial = base.llm_model if provider == base.llm_provider else ""
            st.session_state[custom_key] = initial
        return str(st.session_state.get(custom_key, ""))

    choice_key = f"llm_model_choice_{provider}"
    model_ids = [model.id for model in spec.models]
    valid_choices = {*model_ids, CUSTOM_MODEL}
    if st.session_state.get(choice_key) not in valid_choices:
        initial = base.llm_model if provider == base.llm_provider else spec.default_model
        if initial in model_ids:
            st.session_state[choice_key] = initial
        else:
            st.session_state[choice_key] = CUSTOM_MODEL
            st.session_state.setdefault(custom_key, initial)

    choice = st.session_state.get(choice_key, spec.default_model)
    if choice == CUSTOM_MODEL:
        return str(st.session_state.get(custom_key, ""))
    return str(choice)


def current_settings() -> Settings:
    """Return the environment defaults overlaid with the current GUI values."""
    base = get_settings()
    provider = _initialise_provider(base)
    model = _model_for(provider, base)

    entered_key = str(st.session_state.get(f"llm_api_key_{provider}", ""))
    environment_key = base.api_key if provider == base.llm_provider else None
    api_key = entered_key or environment_key

    entered_url = str(st.session_state.get("llm_custom_base_url", ""))
    environment_url = base.llm_base_url if provider == base.llm_provider else None
    base_url = entered_url or environment_url

    return with_model_access(
        base,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def render(language: Language) -> Settings:
    """Render the complete provider form and return the effective settings."""
    base = get_settings()
    provider = _initialise_provider(base)
    before = current_settings()

    title = _copy(language, "Connect an AI model", "وصّل نموذج ذكاء اصطناعي")
    with st.expander(title, expanded=not before.llm_available):
        provider = st.selectbox(
            _copy(language, "Provider", "المزوّد"),
            options=list(PROVIDERS),
            format_func=lambda key: _provider_label(key, language),
            key="llm_provider",
        )
        spec = PROVIDERS[provider]

        note = spec.access_note
        if language.code == "ar":
            note = (
                "متاح له استخدام مجاني بحدود يحددها المزوّد."
                if spec.free_tier
                else "الاستخدام يُحاسَب عليه من المزوّد مباشرة."
            )
        st.caption(note)

        model_ids = [model.id for model in spec.models]
        labels = {model.id: model.label for model in spec.models}
        custom_key = f"llm_custom_model_{provider}"
        if model_ids:
            choice_key = f"llm_model_choice_{provider}"
            _model_for(provider, base)
            st.selectbox(
                _copy(language, "Model", "النموذج"),
                options=[*model_ids, CUSTOM_MODEL],
                format_func=lambda model: (
                    _copy(language, "Another model ID…", "Model ID مختلف…")
                    if model == CUSTOM_MODEL
                    else labels[model]
                ),
                key=choice_key,
            )
            if st.session_state[choice_key] == CUSTOM_MODEL:
                st.text_input(
                    "Model ID",
                    key=custom_key,
                    placeholder="provider/model-name",
                )
        else:
            _model_for(provider, base)
            st.text_input(
                "Model ID",
                key=custom_key,
                placeholder="your-model-name",
            )

        if spec.requires_base_url:
            st.text_input(
                "Base URL",
                key="llm_custom_base_url",
                placeholder="https://api.example.com/v1",
            )

        st.text_input(
            "API key",
            type="password",
            key=f"llm_api_key_{provider}",
            placeholder=spec.key_variable,
            help=_copy(
                language,
                "The key stays in this app session and is never written to disk.",
                "المفتاح يفضل داخل جلسة التطبيق ولا يُكتب على الجهاز.",
            ),
        )

        environment_key = base.api_key if provider == base.llm_provider else None
        if environment_key and not st.session_state.get(f"llm_api_key_{provider}"):
            st.caption(
                _copy(
                    language,
                    f"Using {spec.key_variable} from the environment.",
                    f"بيتم استخدام {spec.key_variable} من إعدادات البيئة.",
                )
            )
        elif spec.key_url:
            link_label = _copy(language, "Get an API key", "هات API key")
            st.caption(f"[{link_label}]({spec.key_url})")

        settings = current_settings()
        fingerprint = _connection_fingerprint(settings)
        previous = st.session_state.get("llm_connection_result")
        matching = previous if previous and previous[0] == fingerprint else None
        verified = bool(matching and matching[1])

        button_label = _copy(language, "Test connection", "اختبر الاتصال")
        if verified:
            button_label = _copy(language, "Connection verified", "تم تأكيد الاتصال")
        if st.button(
            button_label,
            disabled=not settings.llm_available or verified,
            width="stretch",
            key="test_llm_connection",
        ):
            spinner = _copy(language, "Contacting the provider…", "بنتصل بالمزوّد…")
            with st.spinner(spinner):
                connected, message = ReasoningEngine(settings).probe()
            st.session_state.llm_connection_result = (
                fingerprint,
                connected,
                message,
            )
            matching = st.session_state.llm_connection_result

        if matching and matching[1]:
            st.success(_copy(language, "Connection works.", "الاتصال شغال."))
        elif matching:
            st.error(matching[2])

        if not settings.llm_available and not settings.offline:
            st.info(
                _copy(
                    language,
                    "No valid key yet. The analysis still works with deterministic rules.",
                    "لسه مفيش مفتاح صالح. التحليل هيشتغل بالقواعد الإحصائية العادية.",
                )
            )

        st.caption(
            _copy(
                language,
                "Keys entered here last for this app session only and are never saved with a run.",
                "أي مفتاح تدخله هنا بيفضل للجلسة الحالية بس ومش بيتحفظ مع التحليل.",
            )
        )
        st.caption(
            _copy(
                language,
                "Smart savings is always on: repeated requests reuse the session cache, and a successful connection test is not charged twice.",
                "التوفير الذكي شغال دايمًا: الطلب المتكرر بيستخدم نتيجة الجلسة، واختبار الاتصال الناجح مش بيتحسب مرتين.",
            )
        )

    return current_settings()
