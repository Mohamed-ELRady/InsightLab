"""Language-model access configured from the Streamlit sidebar.

Keys typed here live only in Streamlit session state. They are deliberately not
copied into ``os.environ`` or written to ``.env``: a hosted app may serve more
than one person, and a convenience feature must not turn one person's key into
another person's credential.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from ..core.config import PROVIDERS, ProviderRoute, Settings, get_settings, with_model_access
from ..core.language import Language
from ..core.reasoning import ReasoningEngine
from . import chatgpt_panel

CUSTOM_MODEL = "__custom_model__"


def _connection_fingerprint(settings: Settings) -> str:
    """Identify one exact connection setup without retaining another key copy."""
    parts: list[str] = []
    for route in settings.routes:
        parts.extend((route.provider, route.model, route.base_url or ""))
        parts.extend(route.api_keys)
        parts.append(str(route.chatgpt_connected))
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _copy(language: Language, english: str, arabic: str) -> str:
    return arabic if language.code == "ar" else english


def _check_message(language: Language, message: str) -> str:
    """Translate connection diagnostics without hiding the actionable cause."""
    if language.code != "ar":
        return message
    lower = message.casefold()
    if message == "Connection works.":
        return "الاتصال يعمل."
    if "sign in with chatgpt" in lower:
        return "سجّل الدخول بحساب ChatGPT أولًا."
    if "signed in" in lower:
        return "تم تسجيل الدخول. سيتم التأكد من إتاحة النموذج عند أول طلب تحليل."
    if "api key" in lower and any(word in lower for word in ("missing", "required", "invalid")):
        return "مفتاح API مفقود أو غير صالح. راجع المفتاح وحاول مرة أخرى."
    if "unauthor" in lower or "401" in lower or "403" in lower:
        return "رفض المزوّد بيانات الاتصال. راجع المفتاح وصلاحيات الحساب."
    if "rate" in lower or "429" in lower or "quota" in lower:
        return "وصل الحساب إلى حد الاستخدام أو المعدل المسموح. سيتم تجربة المفتاح أو المزوّد التالي أثناء التحليل."
    if "timeout" in lower or "timed out" in lower:
        return "انتهت مهلة الاتصال بالمزوّد. تحقق من الشبكة وحاول مرة أخرى."
    if "connection failed" in lower:
        detail = message.split(":", 1)[-1].strip()
        return f"فشل الاتصال: {detail}"
    return message


def _provider_label(key: str, language: Language) -> str:
    spec = PROVIDERS[key]
    label = (
        "واجهة متوافقة مع OpenAI"
        if language.code == "ar" and key == "custom"
        else spec.label
    )
    if not spec.free_tier:
        return label
    suffix = "خطة مجانية" if language.code == "ar" else "free tier"
    return f"{label} · {suffix}"


def _matching_result(settings: Settings):
    previous = st.session_state.get("llm_connection_result")
    return (
        previous
        if previous and previous[0] == _connection_fingerprint(settings)
        else None
    )


def _health(
    settings: Settings,
    language: Language,
    reasoning: ReasoningEngine | None = None,
) -> tuple[str, str]:
    """Return the persistent indicator label and its one-line explanation."""
    active_reasoning = reasoning if reasoning and reasoning.settings == settings else None
    matching = _matching_result(settings)

    if settings.offline:
        return (
            _copy(language, "⚪ AI off · offline mode", "⚪ الذكاء الاصطناعي متوقف · وضع بدون اتصال"),
            _copy(
                language,
                "Offline mode is enabled; statistical analysis is still available.",
                "وضع دون اتصال مفعّل؛ التحليل الإحصائي ما زال شغالًا.",
            ),
        )
    if active_reasoning and active_reasoning.configured_route_count and not active_reasoning.available:
        return (
            _copy(language, "🔴 AI unavailable", "🔴 الذكاء الاصطناعي غير متاح"),
            _copy(language, "Every configured route failed. Open to inspect or edit it.", "كل مسارات الاتصال فشلت. افتح اللوحة لمعرفة السبب أو تعديلها."),
        )
    if matching and len(matching) > 3 and matching[3]:
        checks = matching[3]
        working = sum(bool(item["connected"]) for item in checks)
        total = len(checks)
        if working == total:
            return (
                _copy(language, f"🟢 AI working · {working}/{total}", f"🟢 الذكاء الاصطناعي شغال · {working}/{total}"),
                _copy(language, "Every tested connection is working.", "كل الاتصالات التي تم اختبارها تعمل."),
            )
        if working:
            return (
                _copy(language, f"🟠 AI partially working · {working}/{total}", f"🟠 الذكاء الاصطناعي شغال جزئيًا · {working}/{total}"),
                _copy(language, "At least one fallback works; open to see the failed routes.", "يوجد مسار احتياطي واحد على الأقل يعمل؛ افتح لرؤية المسارات التي فشلت."),
            )
        return (
            _copy(language, "🔴 AI connection problem", "🔴 مشكلة في اتصال الذكاء الاصطناعي"),
            _copy(language, "No tested connection is working.", "لا يوجد اتصال تم اختباره ويعمل حاليًا."),
        )
    if active_reasoning and active_reasoning.route_success_count:
        partial = bool(active_reasoning.failover_log)
        return (
            _copy(
                language,
                "🟠 AI working on fallback" if partial else "🟢 AI working",
                "🟠 الذكاء الاصطناعي شغال على مسار احتياطي" if partial else "🟢 الذكاء الاصطناعي شغال",
            ),
            _copy(language, "A real analysis request succeeded.", "تم تنفيذ طلب تحليل فعلي بنجاح."),
        )
    if settings.llm_available:
        return (
            _copy(language, "🟠 AI configured · not tested", "🟠 الذكاء الاصطناعي مضبوط · لم يُختبر"),
            _copy(language, "Open this panel and test all connections.", "افتح اللوحة واختبر كل الاتصالات."),
        )
    return (
        _copy(language, "🔴 AI needs setup", "🔴 الذكاء الاصطناعي يحتاج إعداد"),
        _copy(language, "Add a valid connection; built-in analysis still works.", "أضف اتصالًا صالحًا؛ التحليل المدمج سيظل يعمل."),
    )


def _render_connection_status(
    settings: Settings,
    language: Language,
    reasoning: ReasoningEngine | None,
) -> None:
    """Show active route and per-key state without ever displaying a secret."""
    _, detail = _health(settings, language, reasoning)
    st.caption(detail)
    if reasoning and reasoning.settings == settings and reasoning.active_route_label:
        st.markdown(
            f"**{_copy(language, 'Active now', 'المسار النشط الآن')}:** "
            f"{reasoning.active_route_label}"
        )

    matching = _matching_result(settings)
    if matching and len(matching) > 3:
        for check in matching[3]:
            marker = "🟢" if check["connected"] else "🔴"
            st.caption(
                f"{marker} {check['label']} — "
                f"{_check_message(language, check['message'])}"
            )
        return

    for priority, route in enumerate(settings.routes, start=1):
        issue = Settings._route_issue(route)
        prefix = f"#{priority} · {route.label}"
        if route.provider == "chatgpt":
            marker = "🟠" if route.chatgpt_connected else "🔴"
            message = (
                _copy(language, "ready to verify", "جاهز للاختبار")
                if route.chatgpt_connected else _check_message(language, issue or "")
            )
            st.caption(f"{marker} {prefix} — {message}")
        elif route.api_keys:
            for key_index in range(1, len(route.api_keys) + 1):
                marker = "🟠" if issue is None else "🔴"
                message = (
                    _copy(language, "configured; not tested", "مُضاف ولم يُختبر")
                    if issue is None else _check_message(language, issue)
                )
                st.caption(f"{marker} {prefix} · {_copy(language, 'key', 'مفتاح')} {key_index} — {message}")
        else:
            st.caption(f"🔴 {prefix} — {_check_message(language, issue or 'API key is missing')}")


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


def _key_widget_key(provider: str, index: int) -> str:
    return f"llm_api_key_{provider}" if index == 0 else f"llm_api_key_{provider}_{index + 1}"


def _api_keys_for(provider: str, base: Settings) -> tuple[str, ...]:
    if provider == "chatgpt":
        return ()
    count = int(st.session_state.get(f"llm_api_key_count_{provider}", 1))
    keys = [str(st.session_state.get(_key_widget_key(provider, index), "")).strip()
            for index in range(max(1, count))]
    entered = tuple(dict.fromkeys(key for key in keys if key))
    if entered:
        return entered
    if provider == base.llm_provider:
        return base.api_keys or ((base.api_key,) if base.api_key else ())
    return ()


def _base_url_for(provider: str, base: Settings) -> str | None:
    if not PROVIDERS[provider].requires_base_url:
        return None
    entered = str(st.session_state.get(f"llm_base_url_{provider}", "")).strip()
    legacy = str(st.session_state.get("llm_custom_base_url", "")).strip()
    environment = base.llm_base_url if provider == base.llm_provider else None
    return entered or legacy or environment


def _route_for(provider: str, base: Settings) -> ProviderRoute:
    client = st.session_state.get("chatgpt_client")
    local = st.get_option("server.address") in {"127.0.0.1", "localhost", "::1"}
    return ProviderRoute(
        provider=provider,
        model="auto" if provider == "chatgpt" else _model_for(provider, base),
        api_keys=_api_keys_for(provider, base),
        base_url=_base_url_for(provider, base),
        chatgpt_connected=provider == "chatgpt" and bool(client and client.connected) and local,
    )


def current_settings() -> Settings:
    """Return the environment defaults overlaid with the current GUI values."""
    base = get_settings()
    provider = _initialise_provider(base)
    primary = _route_for(provider, base)
    selected_fallbacks = [
        key for key in st.session_state.get("llm_fallback_providers", [])
        if key in PROVIDERS and key != provider
    ]
    fallbacks = tuple(_route_for(key, base) for key in selected_fallbacks)
    return with_model_access(
        base,
        provider=provider,
        model=primary.model,
        api_key=primary.api_keys[0] if primary.api_keys else None,
        api_keys=primary.api_keys,
        base_url=primary.base_url,
        chatgpt_connected=primary.chatgpt_connected,
        fallback_routes=fallbacks,
    )


def _render_route_fields(provider: str, base: Settings, language: Language) -> None:
    spec = PROVIDERS[provider]
    if provider == "chatgpt":
        chatgpt_panel.render(language)
        return

    note = spec.access_note
    if language.code == "ar":
        note = (
            "متاح له استخدام مجاني بحدود يحددها المزوّد."
            if spec.free_tier else "الاستخدام يُحاسَب عليه من المزوّد مباشرة."
        )
    st.caption(note)

    model_ids = [model.id for model in spec.models]
    labels = {
        model.id: (
            model.label.replace(" · faster", " · أسرع")
            .replace(" · automatic", " · تلقائي")
            if language.code == "ar" else model.label
        )
        for model in spec.models
    }
    custom_key = f"llm_custom_model_{provider}"
    if model_ids:
        choice_key = f"llm_model_choice_{provider}"
        _model_for(provider, base)
        st.selectbox(
            _copy(language, "Model", "النموذج"),
            options=[*model_ids, CUSTOM_MODEL],
            format_func=lambda model: (
                _copy(language, "Another model ID…", "Model ID مختلف…")
                if model == CUSTOM_MODEL else labels[model]
            ),
            key=choice_key,
        )
        if st.session_state[choice_key] == CUSTOM_MODEL:
            st.text_input(_copy(language, "Model ID", "معرّف النموذج"), key=custom_key, placeholder="provider/model-name")
    else:
        _model_for(provider, base)
        st.text_input(_copy(language, "Model ID", "معرّف النموذج"), key=custom_key, placeholder="your-model-name")

    if spec.requires_base_url:
        st.text_input(
            _copy(language, "Base URL", "رابط API الأساسي"), key=f"llm_base_url_{provider}",
            placeholder="https://api.example.com/v1",
        )

    count_key = f"llm_api_key_count_{provider}"
    environment_count = len(base.api_keys) if provider == base.llm_provider else 0
    st.session_state.setdefault(count_key, max(1, environment_count))
    count = st.number_input(
        _copy(language, "Number of API keys", "عدد مفاتيح API"),
        min_value=1, max_value=5, step=1,
        key=count_key,
        help=_copy(
            language,
            "Keys are tried in order before moving to the next provider.",
            "المفاتيح بتتجرّب بالترتيب قبل الانتقال للمزوّد التالي.",
        ),
    )
    for index in range(int(count)):
        label = _copy(language, f"API key {index + 1}", f"مفتاح API رقم {index + 1}")
        st.text_input(
            label, type="password", key=_key_widget_key(provider, index),
            placeholder=spec.key_variable,
            help=_copy(
                language,
                "Stored in this app session only; never written to disk.",
                "المفتاح يفضل داخل جلسة التطبيق فقط ولا يُكتب على الجهاز.",
            ),
        )

    environment_key = base.api_key if provider == base.llm_provider else None
    if environment_key and not st.session_state.get(_key_widget_key(provider, 0)):
        st.caption(_copy(
            language, f"Using {spec.key_variable} from the environment as key 1.",
            f"بيتم استخدام {spec.key_variable} من إعدادات البيئة كمفتاح رقم 1.",
        ))
    elif spec.key_url:
        link_label = _copy(language, "Get an API key", "هات API key")
        st.caption(f"[{link_label}]({spec.key_url})")


def render(
    language: Language,
    reasoning: ReasoningEngine | None = None,
) -> Settings:
    """Render the persistent status/config panel and update a live run."""
    base = get_settings()
    provider = _initialise_provider(base)
    before = current_settings()

    title, _ = _health(before, language, reasoning)
    with st.expander(title, expanded=not before.llm_available):
        _render_connection_status(before, language, reasoning)
        st.divider()
        st.markdown(f"**{_copy(language, 'Edit connections', 'تعديل الاتصالات')}**")
        provider = st.selectbox(
            _copy(language, "Provider", "المزوّد"),
            options=list(PROVIDERS),
            format_func=lambda key: _provider_label(key, language),
            key="llm_provider",
        )
        st.markdown(f"**{_copy(language, 'Primary provider', 'المزوّد الأساسي')}**")
        _render_route_fields(provider, base, language)

        st.divider()
        fallback_options = [key for key in PROVIDERS if key != provider]
        current_fallbacks = [
            key for key in st.session_state.get("llm_fallback_providers", [])
            if key in fallback_options
        ]
        if current_fallbacks != st.session_state.get("llm_fallback_providers", []):
            st.session_state.llm_fallback_providers = current_fallbacks
        fallbacks = st.multiselect(
            _copy(language, "Fallback providers (in order)", "المزوّدون الاحتياطيون (بالترتيب)"),
            options=fallback_options,
            default=current_fallbacks,
            format_func=lambda key: _provider_label(key, language),
            key="llm_fallback_providers",
            placeholder=_copy(language, "Choose providers", "اختر المزوّدين"),
            help=_copy(
                language,
                "After all keys for the primary provider fail, these providers are tried from left to right.",
                "بعد فشل كل مفاتيح المزوّد الأساسي، يتم تجربة المزوّدين الاحتياطيين بالترتيب المحدد.",
            ),
        )
        for priority, fallback in enumerate(fallbacks, start=2):
            with st.expander(f"#{priority} · {_provider_label(fallback, language)}", expanded=True):
                _render_route_fields(fallback, base, language)

        settings = current_settings()
        fingerprint = _connection_fingerprint(settings)
        previous = st.session_state.get("llm_connection_result")
        matching = previous if previous and previous[0] == fingerprint else None
        verified = bool(
            matching and matching[1]
            and (len(matching) < 4 or (matching[3] and all(item["connected"] for item in matching[3])))
        )

        testable_api = any(
            route.provider != "chatgpt" and route.api_keys for route in settings.routes
        )
        if testable_api:
            button_label = _copy(language, "Test all connections", "اختبر كل الاتصالات")
            if verified:
                button_label = _copy(language, "Retest all connections", "أعد اختبار كل الاتصالات")
            if st.button(
                button_label,
                disabled=not settings.llm_available,
                width="stretch",
                key="test_llm_connection",
            ):
                spinner = _copy(language, "Contacting the providers…", "بنتصل بالمزوّدين…")
                with st.spinner(spinner):
                    checks = ReasoningEngine(
                        settings,
                        chatgpt_client=st.session_state.get("chatgpt_client"),
                    ).probe_all()
                    connected = any(check["connected"] for check in checks)
                    working = sum(check["connected"] for check in checks)
                    message = _copy(
                        language,
                        f"{working}/{len(checks)} configured connection(s) work.",
                        f"{working}/{len(checks)} اتصال مضبوط شغال.",
                    )
                st.session_state.llm_connection_result = (
                    fingerprint, connected, message, checks,
                )
                st.rerun()

            if matching and matching[1]:
                st.success(matching[2])
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

    settings = current_settings()
    if reasoning is not None:
        changed = reasoning.reconfigure(
            settings,
            chatgpt_client=st.session_state.get("chatgpt_client"),
        )
        if changed:
            st.toast(_copy(
                language,
                "AI connections updated for this analysis.",
                "تم تحديث اتصالات الذكاء الاصطناعي لهذا التحليل.",
            ))
    return settings
