"""Local-only account controls; all authentication is handled by Codex."""

from __future__ import annotations

import streamlit as st

from ..core.chatgpt import ChatGPTClient, ChatGPTError, find_codex
from ..core.language import Language


def _copy(language: Language, english: str, arabic: str) -> str:
    return arabic if language.code == "ar" else english


def error_message(language: Language, message: str) -> str:
    """Translate safe client diagnostics without showing provider payloads."""
    if language.code != "ar":
        return message
    lower = message.lower()
    if "usage limit" in lower:
        return "وصلت لحد استخدام ChatGPT. استنى لحد ما الرصيد يتجدد، أو اختار مزوّد تاني بنفسك."
    if "too long" in lower:
        return "الطلب أخد وقت أطول من المسموح. قفلنا الاتصال المحلي؛ سجّل دخول من جديد للمحاولة مرة تانية."
    if "connection closed" in lower:
        return "الاتصال بـCodex اتقفل. سجّل دخول من جديد."
    if "could not reach" in lower:
        return "مش قادرين نوصل لـChatGPT. اتأكد من الإنترنت وحاول تاني."
    if "cancelled or expired" in lower:
        return "تسجيل الدخول اتلغى أو انتهت صلاحيته. ابدأ تسجيل الدخول من جديد."
    if "authentication" in lower:
        return "تعذّر التحقق من حساب ChatGPT. سجّل خروج وبعدين ادخل من جديد."
    if "without an answer" in lower:
        return "ChatGPT أنهى الطلب من غير إجابة. حاول من جديد أو اختار مزوّد تاني."
    return "تعذّر إكمال الاتصال بـCodex. اتأكد إن النسخة الرسمية محدثة وإن حسابك يتيح استخدام Codex."


def disconnect() -> None:
    client = st.session_state.pop("chatgpt_client", None)
    try:
        if client:
            client.logout()
    except ChatGPTError:
        pass  # logout also closes the isolated process and drops credentials.
    st.session_state.pop("llm_response_cache", None)


def render(language: Language) -> None:
    st.caption(_copy(language,
        "Use the Codex allowance included with your ChatGPT plan. No API key; not unlimited or guaranteed free access.",
        "استخدم رصيد Codex المتاح في خطة ChatGPT بتاعتك بدون API key. الاستخدام له حدود ومش مجاني أو غير محدود بالضرورة."))
    st.caption(_copy(language,
        "For this local app only—not a general 'Sign in with ChatGPT' service for a public website. Analysis prompts and data excerpts are sent to OpenAI.",
        "متاح للمنصة المحلية على جهازك، مش خدمة دخول عامة لموقع منشور. طلبات التحليل ومقتطفات البيانات بتتبعت لـOpenAI."))

    # Check the server's binding, not a forgeable browser Host header. Public
    # deployments must not expose an account-backed local Codex runtime.
    if st.get_option("server.address") not in {"127.0.0.1", "localhost", "::1"}:
        st.warning(_copy(language,
            "ChatGPT login is disabled on a non-local server. Start Streamlit with --server.address 127.0.0.1 on your own computer.",
            "الدخول معطّل على السيرفر غير المحلي. شغّل Streamlit على جهازك باستخدام --server.address 127.0.0.1."))
        return
    if not find_codex():
        st.info(_copy(language,
            "Install the official Codex CLI, then restart InsightLab. The other providers still work.",
            "ثبّت Codex CLI الرسمي وبعدين أعد تشغيل InsightLab. باقي المزوّدين شغالين عادي."))
        st.code("npm install -g @openai/codex", language="bash")
        st.link_button(_copy(language, "Codex setup", "إعداد Codex"), "https://developers.openai.com/codex/cli/")
        return

    client = st.session_state.get("chatgpt_client")
    connected = bool(client and client.connected)
    if connected:
        st.success(_copy(language, "Signed in with ChatGPT", "تم تسجيل الدخول بحساب ChatGPT"))
        st.text(f"{client.account.get('email', '')} · {client.account.get('planType', '')}")
        st.caption(_copy(language,
            "Account connected. Model availability and remaining allowance are checked when you start an analysis.",
            "الحساب متصل. إتاحة النموذج والرصيد المتبقي بيتحددوا عند بدء طلب التحليل."))
    elif not (client and client.login):
        if st.button(_copy(language, "Sign in with ChatGPT", "سجّل الدخول بحساب ChatGPT"), key="chatgpt_sign_in", width="stretch"):
            try:
                if client:
                    client.close()
                client = ChatGPTClient()
                st.session_state.chatgpt_client = client
                client.start_login()
            except ChatGPTError as error:
                st.error(error_message(language, str(error)))

    if client and client.login:
        st.link_button(_copy(language, "Continue to ChatGPT", "افتح صفحة تسجيل الدخول"), client.login["authUrl"], width="stretch")
        st.info(_copy(language,
            "Finish signing in in your browser on this computer, then click Check sign-in below. Passwords are entered only on OpenAI's page.",
            "كمّل تسجيل الدخول في المتصفح على نفس الجهاز، وبعدين اضغط «افحص تسجيل الدخول». كلمة السر بتدخلها على صفحة OpenAI فقط."))
        if st.button(_copy(language, "Cancel sign-in", "إلغاء تسجيل الدخول"), key="chatgpt_cancel"):
            disconnect()
            st.rerun()

    if client:
        if st.button(_copy(language, "Check sign-in", "افحص تسجيل الدخول"), key="chatgpt_refresh"):
            try:
                client.refresh_account()
                if client.connected:
                    st.rerun()
                elif client.login_error:
                    st.warning(error_message(language, client.login_error))
                else:
                    st.info(_copy(language, "Not signed in yet. Complete the browser step first.", "لسه الدخول مكملش. كمّل خطوة المتصفح الأول."))
            except ChatGPTError as error:
                st.error(error_message(language, str(error)))
        if st.button(_copy(language, "Sign out / disconnect", "تسجيل الخروج / قطع الاتصال"), key="chatgpt_sign_out"):
            disconnect()
            st.rerun()

    st.caption(_copy(language,
        "Codex selects your account's default model. This session has its own in-memory login; it does not reuse or change your desktop/CLI login. Sign out when finished.",
        "Codex بيختار النموذج الافتراضي لحسابك. الدخول خاص بالجلسة وبيتحفظ في الذاكرة فقط، من غير استخدام أو تغيير حساب التطبيق أو الـCLI. سجّل خروج لما تخلص."))
