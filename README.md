# דו"ח מסכם- פרוייקט סופי

##שלב ראשון- בחירת אוסף נתוני רצפים:
סט הנתונים אשר נבחר על ידנו בקפידה מכיל תמליל של 17 הרצאות בנושא כלכלה. הרצאות אלה נלקחו מאתר COURSER.

##שלב שני- תיאור הנתונים:

הנתונים הינם תמלילי הרצאות בנושאים שונים בכלכלה. הנתונים נלקחו מתוך סט הרצאות באתר הקורסים הידוע קורסרה מתוך מגוון נושאים בתחום הכלכלה.
בחרנו בסט נתונים זה מכיוון שהינו מהווה סט נתונים רציף ואשר יהיה ניתן לייצר עבורו רצפים חזויים באמצעות מודל נלמד.
בנוסף, משום שרצינו טקסט מוכוון נושא לשם קבלת תוצאות ענייניות בתחום ספציפי.
מספר המילים הייחודיות הכולל בסט הנתונים הינו 3175
מספר המשפטים הכולל בסט הנתונים הינו 1751  
במהלך העבודה עם נתונים אלו עמדו בפנינו מספר אתגרים:

* ההרצאות בכלכלה כפי שקיבלנו אותם מהאתר הגיעו מפוזרות בקבצי טקסט שונים (כל הרצאה בקובץ טקסט נפרד). 
ע"מ לבצע ניתוח איכותי על הנתונים היה עלינו לבצע איחוד של הנתונים לקובץ טקסט אחד.
* ע"מ לבצע את החיזוי בצורה יעילה היה עלינו להתעלם ממילים ששכיחותן נמוכה(מושגים שכיחים בכלכלה שחוזרים על עצמם לעיתים תכופות בטקסט). נסביר בהמשך מדוע
* ניקוי סימני פיסוק מהטקסט הגולמי והפיכתו למערך של מילים בלבד ע"מ לבצע חיזוי על הטקסט בלבד

מחקר זה שביצענו הינו מחקר מעניין מאוד ובעל פוטנציאל תרומה גדול לעולם האקדמיה. זאת מכיוון שבהינתן סט נתונים היסטורי עשיר מספיק יהיה ניתן לחזות ולייצר הרצאות שלמות בנושא מסוים ללא צורך במשאב אנושי(מרצה).
בנוסף, יהיה ניתן לבצע ניתוחים על אופיים של ההרצאות ולשפרן במידת הצורך.

##שלב שלישי- תיאור שלבי עיבוד מוקדם שהופעלו על הנתונים לטובת הבאתם לפורמט המתאים לשמש קלט לאלגוריתם הלומד:
המחלקה PROCESS:
```{r}
class Process:
    def __init__(self):
        # Assign instance variables
        self.vocabulary_size = 2000
        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"
        self.X_train = None
        self.y_train = None
```
יצרנו את מחלקה זו לצורך ביצוע עיבוד מוקדם.
ביצענו איתחול למשתני המחלקה באופן הבא:
הגדרת גודל הקורפוס שלנו ל2000 ע"מ להשתמש ב2000 המילים הנפוצות בטקסט שלנו
הגדרת משתנים שיהוו לנו אינדקציה לתחילת משפט וסוף משפט
משתנה שיהווה ייצוג למילים בעלות שכיחות נמוכה(נדירות)

```{r}
    def start_process(self,path):
        content = ""
        for fname in glob.glob(path):
            with open(fname, 'r') as content_file:
                content = content + content_file.read()
        self.split_sentence(content)
        return [self.X_train,self.y_train,self.vocabulary_size,self.word_to_index,self.index_to_word]
```
פןנקציה זו קוראת את התוכן מכל מסמכי הטקסט המופרדים לפי הרצאות ומכניסה אותם לתוך משתנה אחד ע"מ שיהוו קובץ טקסט מאוחד

```{r}
    def split_sentence(self,content):
        sentences = content.split('.')
        self.split_words(sentences)
```
פונקציה זו מבצעת פיצול למשפפטים בקובץ הטקטס ע"פ נקודה ויוצרת מערך של משפטים מופרדים.
```{r}
    def split_words(self,sentences):
        sentences = [re.sub('[%s]' % re.escape(string.punctuation), '', sent) for sent in sentences]
        sentences = [sent.replace('\n', ' ') for sent in sentences]
        sentences = ["%s %s %s" % (self.sentence_start_token, x, self.sentence_end_token) for x in sentences]
        words = [nltk.word_tokenize(sent) for sent in sentences]
        self.get_train_set(words)
```
פונקציה זו מבצעת פיצול למילים הנפרדות בכל משפט.
ראשית, הפונקציה משתילה ייצוג לתחילת משפט וסוף משפט לכל משפט, מסירה מכל משפט את סימני הפיסוק שבו ולאחר מכן מבצעת פיצול למילים הנפרדות ויוצרת מערך של כל המילים בטקסט.


```{r}
    def get_train_set(self,words):
        word_freq = nltk.FreqDist(itertools.chain(*words))
        vocab = word_freq.most_common(self.vocabulary_size - 1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(self.unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
        self.index_to_word = index_to_word
        self.word_to_index = word_to_index
        for i, sent in enumerate(words):
            words[i] = [w if w in word_to_index else self.unknown_token for w in sent]
        self.X_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in words])
        self.y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in words])
```
פונקציה זו בונה מילון תדירויות עבור כל מילה, כלומר כמה פעמיםהופיעה כל מילה ייחודית בטקסט כולו.
לאחר מכן, בעזרת המילון שבנינו אנו בוחרים את 2000 המילים השכיחות ביותר בטקסט לשם סיווג טוב יותר.
זאת מכיוון שאם הקורפוס שלנו יכיל מסםר רב של מילים משך זמן הלמידה של המודל יהיה ארוך וממושך יותר. 
בנוסף לכך, למודל יהיה קשה ללמוד על מילים נדירות מכיוון שאין לו הרבה דוגמאות היסטוריות שמכילות אותם ולכן לא הוא לא יוכל ללמוד לגביהם בצורה יעילה.
בפונקציה זו אנחנו יוצרים שני מילוני אינדקס , אחד שמהווה ייצוג מספרי עבור כל מילה ושני שמגדיר לכל מספר איזה מילה הוא מייצג.

x_train מכיל מערך של מערכים כאשר כל מערך בו הוא משפט שהמילים בו מיוצגות ע"פ האינדקס המספרי שלהם
y_train זהה, אך בעל הסטה ימינה לכל מילה במשפט
