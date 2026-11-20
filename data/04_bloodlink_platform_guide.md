---
doc_id: 04_bloodlink_platform_guide
topic: platform_guide
languages: [ar, en]
audience: donor_and_requester
review_status: "DRAFT - verify screen names and flows against the actual frontend before production"
---

<!-- DEV NOTE (strip before embedding): user-facing text only. Do NOT put raw API paths or admin credentials in the chatbot knowledge base. Button/menu names marked [verify] must be checked against the real UI. API mapping for the team:
register=POST /api/auth/register, login=POST /api/auth/login, profile=GET/PUT /api/auth/me|profile,
centers=GET /api/centers, book=POST /api/reservations, my bookings=GET /api/reservations/my,
cancel=PATCH /api/reservations/:id/cancel, my donations=GET /api/donations/my,
blood request=POST /api/blood-requests, notifications=GET /api/notifications -->

# دليل استخدام منصة BloodLink (العربية)

## إيه هي BloodLink؟

BloodLink منصة ويب بتساعد الناس تلاقي أكياس الدم بسرعة. تكتب فصيلة الدم وعدد الأكياس المطلوبة، وتشوف أقرب بنوك الدم والمستشفيات اللي عندها الكمية دي، بدل ما تتصل بكل مكان لوحده. وكمان بتربط المتبرعين بالمراكز والمستشفيات. المنصة بتدعم العربي والإنجليزي.

## المدن المتاحة حالياً

المنصة حالياً في مرحلة تجريبية (Pilot) في **القاهرة والجيزة والإسكندرية**. التوسع لباقي المحافظات مخطط له لاحقاً.

## إزاي تسجل حساب متبرع (مجاني)

1. افتح موقع BloodLink واضغط على **تسجيل حساب جديد** [verify].
2. أدخل بياناتك الأساسية (الاسم، البريد الإلكتروني، رقم الموبايل، فصيلة الدم، المدينة، وكلمة مرور) [verify].
3. أكمل التسجيل، وبكده يبقى عندك حساب متبرع. **التسجيل كمتبرع مجاني.**
4. بعد كده تقدر تسجل دخول في أي وقت وتعدل بياناتك من صفحة **الملف الشخصي**.

## إزاي تسجل دخول وتعدل ملفك الشخصي

سجل دخول بالبريد الإلكتروني وكلمة المرور. من الملف الشخصي تقدر تعدل بياناتك زي رقم الموبايل والمدينة. حافظ على بياناتك محدثة علشان توصلك الإشعارات المناسبة.

## إزاي تحجز موعد تبرع

1. سجل دخول لحسابك.
2. شوف **مراكز التبرع** المتاحة واختار المركز الأنسب ليك [verify].
3. اختار **التاريخ والوقت** المناسبين وأكد الحجز.
4. هتلاقي حجزك في صفحة **حجوزاتي** [verify].
5. روح المركز في ميعادك ومعاك إثبات شخصية، وراجع إرشادات التحضير قبلها.

## إزاي تلغي أو تتابع حجزك

من صفحة حجوزاتك تقدر تشوف حالة كل حجز وتلغيه لو ظروفك اتغيرت. يفضل تلغي بدري علشان المركز يقدر يرتب مواعيد تانية.

## إزاي تتابع سجل تبرعاتك

بعد ما تتبرع وتتسجل التبرعة، بتظهر في **سجل تبرعاتك** [verify]. تقدر تشوف تواريخ تبرعاتك السابقة، وده بيساعدك تعرف إمتى تقدر تتبرع تاني.

## الإشعارات

BloodLink بيبعتلك إشعارات وتذكيرات، زي تأكيد الحجز أو طلبات دم محتاجة متبرعين من فصيلتك قريبة منك. تقدر تشوف كل إشعاراتك وتعلمها كمقروءة من صفحة **الإشعارات** [verify].

## إزاي تطلب دم (لو محتاج كيس دم)

1. سجل دخول لحسابك.
2. اكتب **فصيلة الدم** و**عدد الأكياس** المطلوبة.
3. هتشوف أقرب بنوك الدم والمستشفيات اللي عندها الفصيلة والكمية، مرتبة حسب الأقرب ليك.
4. لو مفيش كفاية متاح، المنصة بتساعد في إيجاد متبرعين مؤهلين قريبين كبديل.
5. ممكن تتابع حالة طلبك من صفحة **طلباتي** [verify].

البحث عن أكياس الدم للمستخدم الطالب خدمة **باشتراك** حسب نموذج المنصة، أما التسجيل كمتبرع فمجاني.

## بطاقة الطوارئ الطبية الرقمية

بطاقة طوارئ رقمية بتحفظ معلوماتك الطبية الأساسية، زي فصيلة الدم، علشان تكون متاحة بسرعة وقت الطوارئ. [verify شكل البطاقة والبيانات اللي بتظهر فيها]

## للمؤسسات

المؤسسات الصحية والخيرية (زي المستشفيات) عندها لوحة تحكم لإدارة الطلبات والمتبرعين. للمؤسسات اشتراك شهري، وبيتم عرض توضيحي (Demo) وتدريب للفريق عند الانضمام.

## لو واجهت مشكلة

لو في مشكلة في الحساب أو الحجز، جرّب تسجل خروج وتدخل تاني، أو تواصل مع فريق الدعم. وفي الحالات الطبية الطارئة اتصل بالإسعاف (123) أو توجه لأقرب مستشفى، ومتعتمدش على الشات بوت.

---

# BloodLink Platform Guide (English)

## What is BloodLink?

BloodLink is a web platform that helps people find blood bags quickly. You enter the blood type and the number of bags you need, and you see the nearest blood banks and hospitals that have that amount, instead of calling each one separately. It also connects donors with centers and hospitals. The platform supports Arabic and English.

## Currently Available Cities

The platform is currently in a pilot phase in **Cairo, Giza, and Alexandria**. Expansion to other governorates is planned for later.

## How to Create a Donor Account (Free)

1. Open the BloodLink website and click **Sign up** [verify].
2. Enter your basic details (name, email, phone number, blood type, city, and a password) [verify].
3. Complete registration, and you now have a donor account. **Registering as a donor is free.**
4. You can then log in anytime and edit your details from the **Profile** page.

## How to Log In and Edit Your Profile

Log in with your email and password. From your profile you can edit details such as your phone number and city. Keep your details up to date so you receive relevant notifications.

## How to Book a Donation Appointment

1. Log in to your account.
2. Browse the available **donation centers** and choose the most convenient one [verify].
3. Choose a suitable **date and time** and confirm the booking.
4. You will find your booking on the **My Reservations** page [verify].
5. Go to the center at your appointment time with a photo ID, and review the preparation guidelines beforehand.

## How to Cancel or Track a Reservation

From your reservations page you can see the status of each booking and cancel it if your plans change. It is best to cancel early so the center can arrange other appointments.

## How to Track Your Donation History

After you donate and the donation is recorded, it appears in your **donation history** [verify]. You can see the dates of your past donations, which helps you know when you can donate again.

## Notifications

BloodLink sends you notifications and reminders, such as booking confirmations or blood requests that need donors of your blood type near you. You can view all notifications and mark them as read from the **Notifications** page [verify].

## How to Request Blood (If You Need a Blood Bag)

1. Log in to your account.
2. Enter the **blood type** and the **number of bags** needed.
3. You will see the nearest blood banks and hospitals that have that type and quantity, sorted by distance.
4. If not enough is available, the platform helps you find eligible donors nearby as a fallback.
5. You can follow your request status on the **My Requests** page [verify].

Searching for blood bags as a requester is a **subscription** service under the platform's model, while registering as a donor is free.

## Digital Emergency Medical Card

A digital emergency card that stores your essential medical information, such as your blood type, so it is quickly available in an emergency. [verify card layout and the fields shown]

## For Organizations

Healthcare and charitable organizations (such as hospitals) have a dashboard for managing requests and donors. Organizations have a monthly subscription, with a live demo and staff training at onboarding.

## If You Face a Problem

If you have an issue with your account or booking, try logging out and back in, or contact the support team. In a medical emergency, call an ambulance (123) or go to the nearest hospital, and do not rely on the chatbot.
