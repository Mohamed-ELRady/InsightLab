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

import re
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
        "en": "You know the context. We uncover the meaning in the data.",
        "ar": "إنت تعرف سياق بياناتك، وإحنا نكشف معناها ونحوّلها لنتيجة تنفع فعلًا.",
    },
    "landing.title": {
        "en": "Understand your own data",
        "ar": "افهم بياناتك بنفسك",
    },
    "landing.intro": {
        "en": (
            "Upload a file and we will work through it together. At every point "
            "where the answer depends on the real-world context rather than "
            "on what the numbers say, we stop and ask you."
        ),
        "ar": (
            "ارفع ملفك وهنشتغل عليه سوا. في كل نقطة الإجابة فيها بتعتمد على طبيعة "
            "مجال البيانات وسياقها مش على الأرقام وحدها، بنقف ونسألك."
        ),
    },
    "landing.files": {"en": "Your data files", "ar": "ملفات بياناتك"},
    "landing.files.help": {
        "en": "CSV, Excel or a tab-separated export. Upload several files and we will work out how they relate. Your originals are never modified; everything is done on a copy.",
        "ar": "ارفع ملف CSV أو Excel أو ملفًا مفصولًا بعلامات تبويب. يمكنك رفع عدة ملفات وسنحدد علاقتها ببعضها. لن نعدّل ملفاتك الأصلية؛ كل العمل يتم على نسخة.",
    },
    "landing.files.multiple": {
        "en": "{count} files. We will ask how each one connects to the largest file before combining them.",
        "ar": "تم اختيار {count} ملفات. سنسألك عن طريقة ربط كل ملف بالملف الأكبر قبل دمجها.",
    },
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
            "knowledge of the domain goes in."
        ),
        "ar": (
            "بنقف ونشرح كل اختيار، وإنت تقرر. دي النقطة اللي معرفتك بمجال البيانات بتدخل "
            "منها."
        ),
    },
    "landing.mode.autonomous.help": {
        "en": (
            "We use our best judgement at every point and you review the "
            "decisions afterwards. Faster, but the analysis knows nothing about "
            "the domain that the file does not say."
        ),
        "ar": (
            "بناخد القرار باجتهادنا في كل نقطة وإنت تراجع بعدين. أسرع، لكن "
            "التحليل مش هيعرف عن مجال البيانات غير اللي ظاهر في الملف."
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
    "landing.deliverable.cleaned": {
        "en": "A cleaned copy of your data, with every change listed",
        "ar": "نسخة منظّفة من بياناتك مع توضيح كل تعديل",
    },
    "landing.deliverable.insights": {
        "en": "The conclusions that matter, each backed by its figure",
        "ar": "الاستنتاجات المهمة، وكل استنتاج مدعوم بأرقامه",
    },
    "landing.deliverable.kpis": {
        "en": "Your headline numbers, each with its calculation",
        "ar": "مؤشراتك الرئيسية مع طريقة حساب كل مؤشر",
    },
    "landing.deliverable.dashboards": {
        "en": "Dashboards arranged for the people who will use them",
        "ar": "لوحات متابعة مرتبة حسب الأشخاص الذين سيستخدمونها",
    },
    "landing.deliverable.reports": {
        "en": "A report in PDF, PowerPoint or Word",
        "ar": "تقرير بصيغة PDF أو PowerPoint أو Word",
    },
    "landing.deliverable.memory": {
        "en": "Everything you tell us, saved for the next analysis",
        "ar": "كل ما تخبرنا به محفوظ للتحليل القادم",
    },
    "landing.areas": {"en": "The areas we can look at", "ar": "المجالات اللي نقدر نحللها"},
    "axis.sales": {"en": "Sales and revenue", "ar": "المبيعات والإيرادات"},
    "axis.customers": {"en": "Customers", "ar": "العملاء"},
    "axis.products": {"en": "Products and services", "ar": "المنتجات والخدمات"},
    "axis.marketing": {"en": "Marketing and channels", "ar": "التسويق وقنوات البيع"},
    "axis.profits": {"en": "Profit and margins", "ar": "الأرباح والهوامش"},
    "axis.regions": {"en": "Regions and locations", "ar": "المناطق والمواقع"},
    "axis.operations": {"en": "Operations and fulfilment", "ar": "العمليات والتنفيذ"},
    "axis.time": {"en": "Change over time", "ar": "التغيّر بمرور الوقت"},
    "axis.comparisons": {"en": "Groups and comparisons", "ar": "المجموعات والمقارنات"},
    "axis.distributions": {"en": "Distributions and unusual values", "ar": "التوزيعات والقيم غير المعتادة"},
    "axis.relationships": {"en": "Relationships between measurements", "ar": "العلاقات بين القياسات"},
    "axis.locations": {"en": "Places and spatial patterns", "ar": "المواقع والأنماط المكانية"},
    "axis.quality": {"en": "Data quality", "ar": "جودة البيانات"},
    "landing.reuse": {
        "en": "Start from approved context learned in earlier analyses",
        "ar": "ابدأ من السياق المعتمد اللي اتعلمناه في تحليلات سابقة",
    },
    "sidebar.progress": {"en": "Progress", "ar": "التقدّم"},
    "sidebar.model": {"en": "Analysis model", "ar": "نموذج التحليل"},
    "sidebar.offline_model": {
        "en": "Offline mode · deterministic analysis",
        "ar": "وضع دون اتصال · تحليل إحصائي حتمي",
    },
    "sidebar.no_credentials": {
        "en": "Without credentials every stage still runs using the built-in statistical rules. Explanations are shorter and use prepared wording rather than a model.",
        "ar": "من غير بيانات اتصال، كل المراحل تظل تعمل بالقواعد الإحصائية المدمجة. الشرح سيكون أقصر وبصياغة جاهزة بدلًا من نموذج ذكاء اصطناعي.",
    },
    "sidebar.memory": {
        "en": "What we know about this project",
        "ar": "اللي نعرفه عن المشروع والبيانات",
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
    "stage.load": {"en": "Loading your data", "ar": "تحميل بياناتك"},
    "stage.understand": {"en": "Understanding the data", "ar": "فهم البيانات"},
    "stage.recall": {"en": "Checking what we already know", "ar": "مراجعة ما نعرفه مسبقًا"},
    "stage.clean": {"en": "Cleaning the data", "ar": "تنظيف البيانات"},
    "stage.features": {"en": "Building new measures", "ar": "إنشاء مقاييس جديدة"},
    "stage.explore": {"en": "Exploring the data", "ar": "استكشاف البيانات"},
    "stage.kpis": {"en": "Summarising performance", "ar": "تلخيص الأداء"},
    "stage.insights": {"en": "Drawing conclusions", "ar": "استخلاص النتائج"},
    "stage.dashboard": {"en": "Building dashboards", "ar": "إنشاء لوحات المتابعة"},
    "stage.report": {"en": "Writing the report", "ar": "إعداد التقرير"},
    "sidebar.api_usage": {"en": "API usage", "ar": "استهلاك الـAPI"},
    "sidebar.api_calls": {
        "en": "{count} model tasks sent in this run.",
        "ar": "تم إرسال {count} مهمة للنموذج في التحليل ده.",
    },
    "sidebar.api_saved": {
        "en": "{count} repeated model tasks served from the session cache.",
        "ar": "تم توفير {count} مهمة متكررة باستخدام نتيجة الجلسة.",
    },
    "sidebar.api_cache_ready": {
        "en": "Session cache is active; repeated work will not use more credit.",
        "ar": "حفظ نتائج الجلسة شغال؛ تكرار نفس الطلب مش هيستهلك كريديت جديد.",
    },
    "sidebar.api_offline": {
        "en": "No model credit is being used.",
        "ar": "مفيش أي كريديت بيتستهلك حاليًا.",
    },
    "sidebar.appearance": {"en": "Appearance", "ar": "المظهر"},
    "sidebar.appearance.light": {"en": "Light", "ar": "فاتح"},
    "sidebar.appearance.dark": {"en": "Dark", "ar": "داكن"},
    "sidebar.appearance.hint": {
        "en": (
            "Set from the ⋮ menu at the top right → Settings → Choose app "
            "theme. It follows your system unless you pick one."
        ),
        "ar": (
            "بيتظبط من قائمة ⋮ فوق على اليمين ← الإعدادات ← اختر سمة التطبيق. "
            "بيتبع سمة جهازك لحد ما تختار واحدة."
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
            "Write in your own words. Anything you say about the data or its domain is "
            "remembered for the rest of the analysis."
        ),
        "ar": (
            "اكتب بكلامك إنت. أي حاجة تقولها عن البيانات أو مجالها بتتحفظ وبتتطبق على باقي "
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
    "understanding.title": {
        "en": "What we understand about this data",
        "ar": "فهمنا إيه عن البيانات دي",
    },
    "understanding.phase1": {
        "en": "Stage 1 · Establishing what the data means before analysis",
        "ar": "المرحلة الأولى · بنفهم البيانات قبل ما نبدأ التحليل",
    },
    "understanding.ready": {
        "en": "Understanding is complete · continuing with the analysis",
        "ar": "اكتمل فهم البيانات · بنكمل دلوقتي باقي التحليل",
    },
    "understanding.domain": {"en": "Detected domain", "ar": "مجال البيانات المكتشف"},
    "understanding.goal": {"en": "Analysis direction", "ar": "اتجاه التحليل"},
    "understanding.questions": {"en": "Questions this data can answer", "ar": "أسئلة تستطيع البيانات الإجابة عنها"},
    "understanding.row": {"en": "One row represents", "ar": "كل صف بيمثل"},
    "understanding.confidence": {"en": "Confidence", "ar": "درجة الثقة"},
    "understanding.answers": {
        "en": "Clarifications you gave us",
        "ar": "التوضيحات اللي قلتها لنا",
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
    "ask.suggestions.thinking": {
        "en": "Working out what is worth asking",
        "ar": "بنحدد أهم الأسئلة المناسبة لبياناتك",
    },
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
    "why.thinking": {"en": "Breaking the movement down", "ar": "بنحلل أسباب التغيّر"},
    "why.unavailable": {
        "en": "Explaining a movement needs a date column and at least three periods of data. This file does not have both.",
        "ar": "تفسير التغيّر يحتاج إلى عمود تاريخ وثلاث فترات زمنية على الأقل، والملف الحالي لا يحتوي على الاثنين.",
    },
    "why.contribution": {"en": "Contribution", "ar": "المساهمة"},
    "why.share": {"en": "Share of the movement", "ar": "نسبة المساهمة في التغيّر"},
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
    "run.working": {
        "en": "Working through your data — loading, cleaning, exploring and writing the report. This usually takes under a minute.",
        "ar": "نعالج بياناتك الآن: تحميل وتنظيف واستكشاف وإعداد التقرير. يستغرق هذا عادةً أقل من دقيقة.",
    },
    "progress.preparing": {"en": "Preparing the analysis", "ar": "بنجهّز التحليل"},
    "progress.saving": {"en": "Saving the uploaded file safely", "ar": "بنحفظ الملف بأمان"},
    "progress.stage.start": {"en": "Starting this stage", "ar": "بنبدأ المرحلة دي"},
    "progress.stage.done": {"en": "Stage complete", "ar": "المرحلة اكتملت"},
    "progress.stage.failed": {"en": "Moving past a stage that could not finish", "ar": "بنتخطى مرحلة ماقدرتش تكتمل"},
    "progress.load.detect": {"en": "Checking the file type and structure", "ar": "بنفحص نوع الملف وتركيبه"},
    "progress.load.read": {"en": "Reading rows and columns", "ar": "بنقرأ الصفوف والأعمدة"},
    "progress.load.tidy": {"en": "Tidying the table structure", "ar": "بنرتّب هيكل الجدول"},
    "progress.load.ready": {"en": "The file is ready for analysis", "ar": "الملف جاهز للتحليل"},
    "progress.understand.profile": {"en": "Profiling columns and data quality", "ar": "بنفحص الأعمدة وجودة البيانات"},
    "progress.understand.meaning": {"en": "Working out what the data represents", "ar": "بنفهم البيانات دي بتمثل إيه"},
    "progress.report.save": {"en": "Saving the cleaned data", "ar": "بنحفظ البيانات المنظّفة"},
    "progress.report.charts": {"en": "Rendering report charts", "ar": "بنجهّز رسومات التقرير"},
    "progress.report.documents": {"en": "Writing the selected documents", "ar": "بننشئ ملفات التقرير"},
    "run.finished": {"en": "Finished", "ar": "اكتمل"},
    "run.found_so_far": {"en": "What we have found so far", "ar": "ما توصلنا إليه حتى الآن"},
    "results.title": {"en": "Analysis of {name}", "ar": "تحليل {name}"},
    "results.project": {"en": "Project: {name}", "ar": "المشروع: {name}"},
    "dashboard.empty": {"en": "No dashboard was built for this run.", "ar": "لم يتم إنشاء لوحة متابعة لهذا التحليل."},
    "dashboard.choose": {"en": "Which view", "ar": "اختر لوحة المتابعة"},
    "insights.empty": {"en": "No conclusion was drawn from this data.", "ar": "لم نتمكن من استخلاص نتيجة موثوقة من هذه البيانات."},
    "insights.about": {
        "en": "Every conclusion includes the figure behind it and its confidence. One file can show that two things move together, but cannot prove that one caused the other.",
        "ar": "كل استنتاج يعرض الرقم الذي يستند إليه ودرجة الثقة فيه. الملف الواحد قد يوضح أن شيئين يتحركان معًا، لكنه لا يثبت أن أحدهما تسبب في الآخر.",
    },
    "insights.chart": {"en": "See the chart behind this", "ar": "اعرض الرسم الداعم للاستنتاج"},
    "kpis.empty": {"en": "No headline figure could be calculated from this data.", "ar": "لم نتمكن من حساب مؤشر رئيسي من هذه البيانات."},
    "kpis.formula": {"en": "How it is calculated", "ar": "طريقة الحساب"},
    "charts.empty": {"en": "No chart was produced.", "ar": "لم يتم إنشاء أي رسم بياني."},
    "charts.areas": {"en": "Areas", "ar": "مجالات التحليل"},
    "data.empty": {"en": "No data is loaded.", "ar": "لا توجد بيانات محمّلة."},
    "data.summary": {"en": "{rows} rows and {columns} columns after cleaning.", "ar": "{rows} صفوف و{columns} أعمدة بعد التنظيف."},
    "data.first_rows": {"en": "Showing the first 500 rows. The download contains all rows.", "ar": "نعرض أول 500 صف. ملف التنزيل يحتوي على كل الصفوف."},
    "data.columns": {"en": "What each column holds", "ar": "محتوى كل عمود"},
    "data.column": {"en": "Column", "ar": "العمود"},
    "data.role": {"en": "Content type", "ar": "نوع المحتوى"},
    "data.empty_values": {"en": "Empty", "ar": "القيم الفارغة"},
    "data.distinct": {"en": "Distinct values", "ar": "القيم المختلفة"},
    "data.notes": {"en": "Notes", "ar": "ملاحظات"},
    "log.changes": {"en": "Every change made to your data", "ar": "كل تعديل تم على بياناتك"},
    "log.no_changes": {"en": "Nothing was changed. The data was analysed exactly as supplied.", "ar": "لم يتم تعديل البيانات؛ تم تحليلها كما رفعتها بالضبط."},
    "log.decisions": {"en": "Every decision taken", "ar": "كل القرارات التي تم اتخاذها"},
    "log.automatic": {"en": "automatic", "ar": "تلقائي"},
    "log.your_choice": {"en": "your choice", "ar": "اختيارك"},
    "log.full": {"en": "The full run log", "ar": "السجل الكامل للتحليل"},
    "log.time": {"en": "Time", "ar": "الوقت"},
    "log.stage": {"en": "Stage", "ar": "المرحلة"},
    "log.event": {"en": "What happened", "ar": "ما حدث"},
    "downloads.empty": {"en": "Nothing was saved for this run.", "ar": "لم يتم حفظ ملفات لهذا التحليل."},
    "downloads.caption": {"en": "These files are also saved on disk, so you can return to them later.", "ar": "هذه الملفات محفوظة أيضًا على الجهاز ويمكنك الرجوع إليها لاحقًا."},
    "downloads.button": {"en": "Download {name}", "ar": "تنزيل {name}"},
    "downloads.memory": {"en": "What we learned about this project", "ar": "ما تعلّمناه عن المشروع والبيانات"},
    "downloads.memory.caption": {"en": "This stays with the project so the next analysis does not ask the same questions again.", "ar": "هذه المعلومات تظل مرتبطة بالمشروع حتى لا يكرر التحليل القادم نفس الأسئلة."},
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


def stage_title(stage: str, language: Language | str = DEFAULT) -> str:
    """Return a translated workflow-stage title without translating stage IDs."""
    return translate(f"stage.{stage}", language)


def axis_label(axis: str, language: Language | str = DEFAULT) -> str:
    """Return a translated presentation label for a stable analysis-axis ID."""
    return translate(f"axis.{axis}", language)


KPI_NAMES_AR = {
    "Total revenue": "إجمالي الإيرادات",
    "Total profit": "إجمالي الأرباح",
    "Profit margin": "هامش الربح",
    "Average order value": "متوسط قيمة الطلب",
    "Growth rate": "معدل النمو",
    "Strongest month": "أقوى شهر",
    "Active customers": "العملاء النشطون",
    "Repeat customer rate": "معدل تكرار العملاء",
    "Top 10% customer share": "حصة أكبر 10% من العملاء",
    "Units sold": "الوحدات المباعة",
    "Number of records": "عدد السجلات",
}

ROLE_NAMES_AR = {
    "identifier": "معرّف",
    "datetime": "تاريخ ووقت",
    "measure": "رقم قابل للقياس",
    "category": "فئة",
    "boolean": "نعم / لا",
    "text": "نص",
    "constant": "قيمة ثابتة",
    "unknown": "غير محدد",
}

ARTEFACT_NAMES_AR = {
    "Cleaned data": "البيانات المنظّفة",
    "Business memory": "ذاكرة المشروع",
    "JSON summary": "ملخص JSON",
    "Run summary": "ملخص التحليل",
    "PDF report": "تقرير PDF",
    "PDF document": "تقرير PDF",
    "PowerPoint report": "عرض PowerPoint",
    "PowerPoint deck": "عرض PowerPoint",
    "Word report": "تقرير Word",
    "Word document": "تقرير Word",
}


def kpi_name(name: str, language: Language | str = DEFAULT) -> str:
    code = language.code if isinstance(language, Language) else str(language)
    return KPI_NAMES_AR.get(name, name) if code == "ar" else name


def role_name(role: str, language: Language | str = DEFAULT) -> str:
    code = language.code if isinstance(language, Language) else str(language)
    return ROLE_NAMES_AR.get(role, role) if code == "ar" else role


def artefact_name(name: str, language: Language | str = DEFAULT) -> str:
    code = language.code if isinstance(language, Language) else str(language)
    if code == "ar":
        return ARTEFACT_NAMES_AR.get(name, name)
    return "project memory" if name == "Business memory" else name.lower()


def kpi_formula(formula: str, language: Language | str = DEFAULT) -> str:
    """Translate known deterministic formula wording and preserve column IDs."""
    code = language.code if isinstance(language, Language) else str(language)
    if code != "ar":
        return formula
    patterns = (
        (r"^Sum of every value in (.+)$", r"مجموع كل القيم في \1"),
        (r"^Sum of (.+)$", r"مجموع \1"),
        (r"^Count of rows after cleaning$", r"عدد الصفوف بعد التنظيف"),
        (r"^Count of distinct values in (.+)$", r"عدد القيم المختلفة في \1"),
        (r"^Share of values in (.+) that appear more than once$", r"نسبة قيم \1 التي تظهر أكثر من مرة"),
        (r"^Total (.+) divided by total (.+), as a percentage$", r"إجمالي \1 ÷ إجمالي \2 كنسبة مئوية"),
        (r"^Total (.+) divided by total (.+)$", r"إجمالي \1 ÷ إجمالي \2"),
        (r"^Total (.+) divided by the number of rows$", r"إجمالي \1 ÷ عدد الصفوف"),
        (r"^Month with the highest total (.+)$", r"الشهر صاحب أعلى إجمالي لـ\1"),
        (r"^Average month in the second half of the period compared with the average month in the first half$", r"متوسط الشهر في النصف الثاني مقارنة بمتوسط الشهر في النصف الأول"),
    )
    for pattern, replacement in patterns:
        if re.match(pattern, formula):
            return re.sub(pattern, replacement, formula)
    return formula


def kpi_meaning(name: str, original: str, language: Language | str = DEFAULT) -> str:
    """Return concise Arabic guidance for built-in KPIs."""
    code = language.code if isinstance(language, Language) else str(language)
    if code != "ar":
        return original
    meanings = {
        "Total revenue": "إجمالي ما حققه النشاط قبل خصم التكاليف.",
        "Total profit": "المبلغ الذي احتفظ به النشاط بعد التكاليف المسجلة.",
        "Profit margin": "النسبة التي يحتفظ بها النشاط من كل 100 من الإيرادات.",
        "Average order value": "متوسط قيمة العملية الواحدة؛ قارنه بالقيمة المعتادة حتى لا ترفعه صفقات كبيرة قليلة.",
        "Growth rate": "يقارن الأداء المعتاد في النصف الأحدث من الفترة بالنصف الأقدم.",
        "Strongest month": "الشهر صاحب أعلى إجمالي، ويساعد في تخطيط المخزون والموارد.",
        "Active customers": "عدد العملاء المختلفين الموجودين في البيانات.",
        "Repeat customer rate": "نسبة العملاء الذين عادوا ونفذوا أكثر من عملية.",
        "Top 10% customer share": "يوضح مدى اعتماد النشاط على أكبر العملاء ومخاطر فقدان أحدهم.",
        "Units sold": "حجم المبيعات الفعلي؛ اقرأه مع الإيرادات للتمييز بين نمو الكمية وارتفاع السعر.",
        "Number of records": "عدد الصفوف التي بُنيت عليها كل الأرقام بعد التنظيف.",
    }
    return meanings.get(name, original)


def profile_note(note: str, language: Language | str = DEFAULT) -> str:
    """Translate profiler-generated notes while preserving their figures."""
    code = language.code if isinstance(language, Language) else str(language)
    if code != "ar" or not note:
        return note
    parts = []
    for part in note.split("; "):
        if match := re.match(r"over half the rows have no value \((.+) empty\)", part):
            parts.append(f"أكثر من نصف الصفوف فارغة ({match.group(1)})")
        elif match := re.match(r"(.+) of rows are empty", part):
            parts.append(f"{match.group(1)} من الصفوف فارغة")
        elif match := re.match(r"(.+) rows hold a negative number", part):
            parts.append(f"{match.group(1)} صفوف تحتوي على رقم سالب")
        elif part == "every row holds the same value, so it cannot explain anything":
            parts.append("كل الصفوف تحمل نفس القيمة، لذلك العمود لا يفسر اختلافًا")
        elif part == "a few very large values pull the average up":
            parts.append("عدد قليل من القيم الكبيرة يرفع المتوسط")
        elif part == "almost every row falls into a single group":
            parts.append("تقريبًا كل الصفوف تقع في مجموعة واحدة")
        else:
            parts.append(part)
    return "؛ ".join(parts)


def answer_description(answer, language: Language | str = DEFAULT) -> str:
    """Localise an answer summary without changing its stored decision payload."""
    code = language.code if isinstance(language, Language) else str(language)
    if code != "ar":
        return answer.describe()
    choice = getattr(answer.choice, "value", str(answer.choice))
    if choice == "skip":
        return "تم تخطي الخطوة."
    if choice == "custom":
        return f"تعليمات مخصصة: {answer.text}"
    selected = answer.selected
    label = selected.label if selected else "اختيار غير معروف"
    if choice == "suggestion":
        return f"تم قبول الاقتراح: {label}"
    return f"تم اختيار البديل: {label}"
