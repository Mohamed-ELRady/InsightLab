"""What language the product speaks to this user.

The product is aimed at business owners who are not data specialists. In much
of the world - and certainly in the region this was built for - most of them
work in Arabic. An English-only interface does not inconvenience that user; it
excludes them entirely.

Two things change with the language and one does not. The interface strings
change, and so does the prose every agent writes, since an explanation is
useless in a language the reader does not use. Column names, log messages and
the code itself stay in English: they are the data's own vocabulary and the
engineer's, and translating them would break every lookup in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    english_name: str
    rtl: bool
    #: How to instruct an agent to write in this language.
    instruction: str


ENGLISH = Language(
    code="en",
    name="English",
    english_name="English",
    rtl=False,
    instruction="",
)

ARABIC = Language(
    code="ar",
    name="العربية",
    english_name="Arabic",
    rtl=True,
    instruction=(
        "Write everything the owner reads in Modern Standard Arabic, in a "
        "natural business register rather than a literary or translated one. "
        "Keep column names, and any value copied from the data, exactly as they "
        "appear in the file - do not translate them, because the owner needs to "
        "find them in their own spreadsheet. Numbers stay in Western digits."
    ),
)

LANGUAGES: dict[str, Language] = {ENGLISH.code: ENGLISH, ARABIC.code: ARABIC}

DEFAULT = ENGLISH


def get(code: str | None) -> Language:
    return LANGUAGES.get((code or "").casefold(), DEFAULT)


#: Interface strings. English is the source; a missing Arabic entry falls back
#: to it rather than showing a key, so a partial translation degrades quietly.
STRINGS: dict[str, dict[str, str]] = {
    "app.tagline": {
        "en": "You know your business. We know the numbers. Between the two we get to something useful.",
        "ar": "إنت خبير شغلك، وإحنا خبراء الأرقام. من التعاون بين الاتنين بنوصل لنتيجة تنفع فعلاً.",
    },
    "landing.title": {
        "en": "Understand your own data",
        "ar": "افهم بياناتك بنفسك",
    },
    "landing.intro": {
        "en": (
            "Upload a file and we will work through it together. At every point "
            "where the answer depends on how *your* business works rather than "
            "on what the numbers say, we stop and ask you."
        ),
        "ar": (
            "ارفع ملفك وهنشتغل عليه سوا. في كل نقطة الإجابة فيها بتعتمد على طبيعة "
            "شغلك إنت مش على الأرقام، بنقف ونسألك."
        ),
    },
    "landing.files": {"en": "Your data files", "ar": "ملفات بياناتك"},
    "landing.mode": {
        "en": "How would you like to work?",
        "ar": "تحب نشتغل إزاي؟",
    },
    "landing.mode.interactive": {
        "en": "Ask me at every decision",
        "ar": "اسألني عند كل قرار",
    },
    "landing.mode.autonomous": {
        "en": "Run it all automatically",
        "ar": "شغّل كل حاجة أوتوماتيك",
    },
    "landing.mode.interactive.help": {
        "en": (
            "We stop and explain each choice, and you decide. This is where your "
            "knowledge of the business goes in."
        ),
        "ar": (
            "بنقف ونشرح كل اختيار، وإنت تقرر. دي النقطة اللي معرفتك بشغلك بتدخل "
            "منها."
        ),
    },
    "landing.mode.autonomous.help": {
        "en": (
            "We use our best judgement at every point and you review the "
            "decisions afterwards. Faster, but the analysis knows nothing about "
            "your business that the file does not say."
        ),
        "ar": (
            "بناخد القرار باجتهادنا في كل نقطة وإنت تراجع بعدين. أسرع، لكن "
            "التحليل مش هيعرف عن شغلك غير اللي في الملف."
        ),
    },
    "landing.start": {"en": "Start the analysis", "ar": "ابدأ التحليل"},
    "landing.sample": {
        "en": "Try it with sample data",
        "ar": "جرّب ببيانات تجريبية",
    },
    "landing.deliverables": {
        "en": "What you will end up with",
        "ar": "هتخرج بإيه",
    },
    "landing.areas": {"en": "The areas we can look at", "ar": "المجالات اللي نقدر نحللها"},
    "landing.reuse": {
        "en": "Start from what a previous analysis already knows about my business",
        "ar": "ابدأ من اللي تحليل سابق عرفه عن شغلي",
    },
    "sidebar.progress": {"en": "Progress", "ar": "التقدّم"},
    "sidebar.model": {"en": "Analysis model", "ar": "نموذج التحليل"},
    "sidebar.memory": {
        "en": "What we know about your business",
        "ar": "اللي نعرفه عن شغلك",
    },
    "sidebar.memory.empty": {
        "en": (
            "Nothing yet. Anything you tell us during the analysis is kept here "
            "and applied to every later stage."
        ),
        "ar": (
            "لسه مفيش حاجة. أي حاجة تقولهالنا أثناء التحليل بتتحفظ هنا وبتتطبق "
            "على كل الخطوات اللي بعدها."
        ),
    },
    "sidebar.memory.add": {
        "en": "Add something we should know",
        "ar": "ضيف حاجة المفروض نعرفها",
    },
    "sidebar.memory.save": {"en": "Remember this", "ar": "افتكر ده"},
    "sidebar.memory.placeholder": {
        "en": "For example: our peak season starts in November.",
        "ar": "مثلاً: موسم الذروة عندنا بيبدأ في نوفمبر.",
    },
    "sidebar.new_run": {"en": "Start a new analysis", "ar": "ابدأ تحليل جديد"},
    "sidebar.earlier": {"en": "Earlier analyses", "ar": "تحليلات سابقة"},
    "sidebar.appearance": {"en": "Appearance", "ar": "المظهر"},
    "sidebar.appearance.light": {"en": "Light", "ar": "فاتح"},
    "sidebar.appearance.dark": {"en": "Dark", "ar": "داكن"},
    "sidebar.appearance.hint": {
        "en": (
            "Set from the ⋮ menu at the top right → Settings → Choose app "
            "theme. It follows your system unless you pick one."
        ),
        "ar": (
            "بيتظبط من قائمة ⋮ فوق على اليمين ← Settings ← Choose app theme. "
            "بيتبع جهازك لحد ما تختار واحد."
        ),
    },
    "decision.prompt": {
        "en": "What would you like to do?",
        "ar": "تحب نعمل إيه؟",
    },
    "decision.recommended": {"en": "recommended", "ar": "مُوصى به"},
    "decision.custom": {
        "en": "Tell us how you want this handled",
        "ar": "قولنا إنت عايز نتصرف إزاي",
    },
    "decision.skip": {"en": "Skip this step", "ar": "تخطَّ الخطوة دي"},
    "decision.continue": {"en": "Continue", "ar": "كمّل"},
    "decision.your_instruction": {"en": "Your instruction", "ar": "تعليماتك"},
    "decision.custom_placeholder": {
        "en": (
            "Write in your own words. Anything you say about your business is "
            "remembered for the rest of the analysis."
        ),
        "ar": (
            "اكتب بكلامك إنت. أي حاجة تقولها عن شغلك بتتحفظ وبتتطبق على باقي "
            "التحليل."
        ),
    },
    "decision.custom_required": {
        "en": "Write your instruction first, or choose one of the other options.",
        "ar": "اكتب تعليماتك الأول، أو اختر واحد من الخيارات التانية.",
    },
    "decision.evidence": {
        "en": "Show the underlying figures",
        "ar": "اعرض الأرقام اللي وراها",
    },
    "tab.ask": {"en": "Ask a question", "ar": "اسأل سؤال"},
    "tab.dashboards": {"en": "Dashboards", "ar": "لوحات المتابعة"},
    "tab.insights": {"en": "What the data shows", "ar": "اللي البيانات بتقوله"},
    "tab.kpis": {"en": "Headline figures", "ar": "الأرقام الرئيسية"},
    "tab.charts": {"en": "All charts", "ar": "كل الرسومات"},
    "tab.data": {"en": "Your data", "ar": "بياناتك"},
    "tab.log": {"en": "What we changed", "ar": "اللي غيّرناه"},
    "tab.downloads": {"en": "Downloads", "ar": "التحميلات"},
    "ask.title": {"en": "Ask about your data", "ar": "اسأل عن بياناتك"},
    "ask.caption": {
        "en": (
            "We can only answer from what is in your file. Anything we cannot "
            "work out, we will say so rather than guess."
        ),
        "ar": (
            "بنجاوب من اللي في ملفك بس. أي حاجة مش قادرين نطلعها، هنقول مش "
            "عارفين بدل ما نخمّن."
        ),
    },
    "ask.placeholder": {
        "en": "Ask a question about your data",
        "ar": "اكتب سؤالك عن بياناتك",
    },
    "ask.suggestions": {
        "en": "Or start with one of these:",
        "ar": "أو ابدأ بواحد من دول:",
    },
    "ask.understood": {"en": "Worked out as", "ar": "فهمناه كـ"},
    "ask.numbers": {"en": "The numbers", "ar": "الأرقام"},
    "ask.thinking": {"en": "Working it out", "ar": "بنحسبها"},
    "why.title": {"en": "Why did that happen?", "ar": "ليه ده حصل؟"},
    "why.breakdown": {"en": "Broken down by", "ar": "التفصيل حسب"},
    "why.exact": {
        "en": (
            "These add up exactly to the total movement, so the shares can be "
            "read as the whole story rather than a sample of it."
        ),
        "ar": (
            "دي بتجمع بالظبط على إجمالي الحركة، يعني النسب دي القصة كاملة مش "
            "عينة منها."
        ),
    },
    "insight.evidence": {"en": "Evidence", "ar": "الدليل"},
    "insight.meaning": {"en": "What this means", "ar": "يعني إيه"},
    "insight.action": {"en": "What to do", "ar": "اعمل إيه"},
    "insight.caveat": {"en": "Read this carefully", "ar": "خد بالك"},
    "insight.objection": {"en": "On review", "ar": "عند المراجعة"},
    "confidence.high": {"en": "Strong evidence", "ar": "دليل قوي"},
    "confidence.medium": {"en": "Worth checking", "ar": "يستاهل التأكد"},
    "confidence.low": {"en": "Treat as a hint", "ar": "مجرد مؤشر"},
    "chart.numbers": {
        "en": "See the numbers behind this chart",
        "ar": "شوف الأرقام اللي ورا الرسم ده",
    },
    "complete": {"en": "Analysis complete", "ar": "التحليل خلص"},
}


def translate(key: str, language: Language | str = DEFAULT) -> str:
    """Look up an interface string.

    Falls back to English when a translation is missing, and to the key itself
    only when the string is unknown - which shows up loudly in review rather
    than silently rendering nothing.
    """
    code = language.code if isinstance(language, Language) else str(language)
    entry = STRINGS.get(key)
    if entry is None:
        return key
    return entry.get(code) or entry.get("en") or key
