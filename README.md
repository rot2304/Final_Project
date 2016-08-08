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
        self.unused_token = "UNKNOWN_TOKEN"
        self.sentence_start = "SENTENCE_START"
        self.sentence_end = "SENTENCE_END"
        self.train_set = None
        self.shifted_train = None

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
        return [self.train_set, self.shifted_train, self.vocabulary_size, self.word_index_dictionary, self.index_word_dictionary]

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
        sentences = ["%s %s %s" % (self.sentence_start, x, self.sentence_end) for x in sentences]
        words = [nltk.word_tokenize(sent) for sent in sentences]
        self.get_train_set(words)

```
פונקציה זו מבצעת פיצול למילים הנפרדות בכל משפט.
ראשית, הפונקציה משתילה ייצוג לתחילת משפט וסוף משפט לכל משפט, מסירה מכל משפט את סימני הפיסוק שבו ולאחר מכן מבצעת פיצול למילים הנפרדות ויוצרת מערך של כל המילים בטקסט.


```{r}
    def get_train_set(self,words):
        word_freq = nltk.FreqDist(itertools.chain(*words))
        vocab = word_freq.most_common(self.vocabulary_size - 1)
        index_word_dictionary = [x[0] for x in vocab]
        index_word_dictionary.append(self.unused_token)
        word_index_dictionary = dict([(w, i) for i, w in enumerate(index_word_dictionary)])
        self.index_word_dictionary = index_word_dictionary
        self.word_index_dictionary = word_index_dictionary
        for i, sent in enumerate(words):
            words[i] = [w if w in word_index_dictionary else self.unused_token for w in sent]
        self.train_set = numpy.asarray([[word_index_dictionary[w] for w in sent[:-1]] for sent in words])
        self.shifted_train = numpy.asarray([[word_index_dictionary[w] for w in sent[1:]] for sent in words])

```
פונקציה זו בונה מילון תדירויות עבור כל מילה, כלומר כמה פעמיםהופיעה כל מילה ייחודית בטקסט כולו.
לאחר מכן, בעזרת המילון שבנינו אנו בוחרים את 2000 המילים השכיחות ביותר בטקסט לשם סיווג טוב יותר.
זאת מכיוון שאם הקורפוס שלנו יכיל מסםר רב של מילים משך זמן הלמידה של המודל יהיה ארוך וממושך יותר. 
בנוסף לכך, למודל יהיה קשה ללמוד על מילים נדירות מכיוון שאין לו הרבה דוגמאות היסטוריות שמכילות אותם ולכן לא הוא לא יוכל ללמוד לגביהם בצורה יעילה.
בפונקציה זו אנחנו יוצרים שני מילוני אינדקס , אחד שמהווה ייצוג מספרי עבור כל מילה ושני שמגדיר לכל מספר איזה מילה הוא מייצג.

x_train מכיל מערך של מערכים כאשר כל מערך בו הוא משפט שהמילים בו מיוצגות ע"פ האינדקס המספרי שלהם
y_train זהה, אך בעל הסטה ימינה לכל מילה במשפט

##שלב רביעי- כתיבת קוד פייתון לאלגוריתם RNN 
באמצעותו יופק מודל לשחזור הנתונים:

המחלקה RNNNumpy:

```{r}
class RNNNumpy:
    def __init__(self, wordDimention, hiddenDimention=100, bptt_truncate=4):
        # Assign instance variables
        self.npa = numpy.array
        self.wordDimention = wordDimention
        self.hiddenDimention = hiddenDimention
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = numpy.random.uniform(-numpy.sqrt(1. / wordDimention), numpy.sqrt(1. / wordDimention), (hiddenDimention, wordDimention))
        self.V = numpy.random.uniform(-numpy.sqrt(1. / hiddenDimention), numpy.sqrt(1. / hiddenDimention), (wordDimention, hiddenDimention))
        self.W = numpy.random.uniform(-numpy.sqrt(1. / hiddenDimention), numpy.sqrt(1. / hiddenDimention), (hiddenDimention, hiddenDimention))

```
יצרנו מחלקה זו ע"מ לבנות את מודל רשת הנוירונים. 
לצורך בניית המודל, בחרנו להשתמש ב100 שכבות ואתחלנו בצורה רנדומלית את הוקטורים U,V,W.

```{r}
       def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = numpy.zeros((T + 1, self.hiddenDimention))
        s[-1] = numpy.zeros(self.hiddenDimention)
        # The outputs at each time step. Again, we save them for later.
        o = numpy.zeros((T, self.wordDimention))
        # For each time step...
        for t in numpy.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = numpy.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = self.softmax(self.V.dot(s[t]))

        return [o, s]

        RNNNumpy.forward_propagation = forward_propagation

```
פונקציה זו מחזירה לנו שני מערכים. הראשון הוא מערך של מערכים שכל מערך בו מייצג את ההסתברות של כל מילה במאגר המילים שלנו להופיע אחרי המילה הספיציפית במשפט שנשלח כקלט לפונקציה. המערך השני מייצג את המצבים החבויים ויהווה לנו לעזר בשלב מאוחר יותר לצורך חישוב הגדיאנטים U,V,W.
```{r}
    def softmax(self,w, t=1.0):
        e = numpy.exp(self.npa(w) / t)
        dist = e / numpy.sum(e)
        return dist
        RNNNumpy.softmax = softmax

```
לשם חישוב מערך ההסתבוריות אנו נדרשים להשתמש בפונקציה זו ע"מ לחשב את ההסתברות
```{r}
      def predict(self, x):
       # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return numpy.argmax(o, axis=1)

        RNNNumpy.predict = predict
```
פונציה זו מקבלת משפט ומחזירה עבור כל מילה במשפט את האינדקס של המילה בעלת ההסתברות הכי גבוהה להופיע אחרי.
```{r}
     def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in numpy.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[numpy.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * numpy.sum(numpy.log(correct_word_predictions))
        return L
```

```{r}
    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = numpy.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N
```
פונקציה זו נועדה למדוד את רמת השגיאות במודל . זאת ע"י השוואת החיזוי שהופק ע"י המודל לעומת טקסט המקור, זאת ע"מ לשפר ולאמן את המודל בתהליך איטרטיבי וע"מ לשפר את הגראדינטים (המשקולות).
```{r}
   def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        U_weight = numpy.zeros(self.U.shape)
        V_weight = numpy.zeros(self.V.shape)
        W_weight = numpy.zeros(self.W.shape)
        delta_o = o
        delta_o[numpy.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in numpy.arange(T)[::-1]:
            V_weight += numpy.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in numpy.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                W_weight += numpy.outer(delta_t, s[bptt_step - 1])
                U_weight[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [U_weight, V_weight, W_weight]

```

ע"מ לחשב את הגראדינטים אשר ישמשו אותנו בפונקציה הבאה נשתמש  בפונקציה זו. 
הגראדינטים אינם מחושבים רק לרגע הנוכחי אלא בהתחשבות בעבר.

```{r}
    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        U_weight, V_weight, W_weight = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * U_weight
        self.V -= learning_rate * V_weight
        self.W -= learning_rate * W_weight

```
פונקציה זו עוברת באיטרציות על כל סט האימון ובכל איטרציה היא מכוונת את הפרמטרים להפחתת הטעות. אנו מעדכנים את הפרמטרים לכיוון שיפחית את הטעות. הכיוונים ניתנים לנו ע"י הגראדיאנטים ופונקציית ההפסד.
בנוסף, ישנו משתנה אשר מהוה את קצב הלמידה שמגדיר עבורנו את גודל הצעד שאנו מבצעים כל איטרציה. יורחב עליו במהשך.

##שלב חמישי- כתיבת קוד לחילול נתוני רצפים ע"פ המודל שנלמד ובניית מודל והרצתו תוך שימוש בו לטובת חילול מידע:
המחלקה classifier:
```{r}
class classifier:
    def __init__(self, vocabulary_size, x_train, y_train):
        numpy.random.seed(10)
        model = RNN_Algorithem.RNNNumpy(vocabulary_size)
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.vocabulary_size = vocabulary_size
        self.learning_rate = 0.005
        self.nepoch = 50
        self.evaluate_loss_after = 1
        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"
```
יצרנו מחלקה זו לצורך בניית מודל רשת הנוירונים שלנו.
לצורך כך הכנסנו את כל הפרטמרים הנדרשים- קצב למידה, גודל הקורפוס, סט האימון המהווה קלט למודל.
```{r}
    def train_with_sgd(self):
            # We keep track of the losses so we can plot them later
            learning_rate = self.learning_rate
            losses = []
            num_examples_seen = 0
            for epoch in range(self.nepoch):
                # Optionally evaluate the loss
                if (epoch % self.evaluate_loss_after == 0):
                    loss = self.model.calculate_loss(self.x_train, self.y_train)
                    losses.append((num_examples_seen, loss))
                    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print( "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))

                    # Adjust the learning rate if loss increases
                    if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                        learning_rate = learning_rate * 0.5
                        print("Setting learning rate to %f" % learning_rate)
                    sys.stdout.flush()
                # For each training example...
                for i in range(len(self.y_train)):
                    # One SGD step
                    self.model.sgd_step(self.x_train[i], self.y_train[i], self.learning_rate)
                    num_examples_seen += 1
```
פונקציה זו מהווה לולאה חיצת אשר עוברת בצורה איטרטיבית על סט האימון בעזרת פונקציית העזר ע"מ לבנות את רשת הנוירונים ולהתאים את קצב הלמידה.

```{r}
    def generate_sentence(self, word_to_index, index_to_word):
        # We start the sentence with the start token
        new_sentence = [word_to_index[self.sentence_start_token]]
        # Repeat until we get an end token
        while not new_sentence[-1] == word_to_index[self.sentence_end_token] :

            next_word_probs = self.model.forward_propagation(new_sentence)
            sampled_word = word_to_index[self.unknown_token]
            # We don't want to sample unknown words
            while sampled_word == word_to_index[self.unknown_token]:

                samples = numpy.random.multinomial(1, next_word_probs[0][-1])
                sampled_word = numpy.argmax(samples)
            new_sentence.append(sampled_word)
        sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]

        return sentence_str
```
פונקציה זו אחראית על משפט בודד. נתחיל כל משפט ע"י הטוקן שמסמן לנו את תחילת משפט ונבצע כל  פעם חיזוי לגבי המילה הבאה ונצרף אותה למשפט. כאשר החיזוי יחזיר לנו כמילה הבאה את המילה שמסמלת סוף משפט , נסיים לחזות ונחזיר את המשפט שהצטבר עד עכשיו.

```{r}
    def start_generate(self,word_to_index, index_to_word):
     num_sentences = 100
     senten_min_length = 7
     text =""
     for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = self.generate_sentence(word_to_index,index_to_word)
        #print(" ".join(sent))
        text = text + " ".join(sent) + "." + "\n"
     with open("Output.txt", "w") as text_file:
      text_file.write(text)
```
פונקציה זו קובעת כמה משפטים נחזה, אורך מינימלי של משפט וקוראת לפונקציית החיזוי "ייצור משפט" כמספר המשפטים שאותם נרצה לחזות ומדפיסה אותם לקובץ טקטסט חיצוני שאותו היא יוצרת
קובץ טקסט זה מהווה בעצם את חילול נתוני הרצפים ע"פ המודל שנלמד.


##שלב שישי- הערכת איכות המידע המשוחזר באמצעות השוואת הרצפים המסונתזים לרצפי המקור (כל רצף מסונתז יושווה לרצף הכי דומה לו בסט המקורי, לבסוף יחושב ממוצע). השתמשו במדד דמיון מתאים לטובת המשימה.  הערכת איכות המידע המשוחזר באמצעות השוואת הרצפים המסונתזים לרצפי המקור (כל רצף מסונתז יושווה לרצף הכי דומה לו בסט המקורי, לבסוף יחושב ממוצע). השתמשו במדד דמיון מתאים לטובת המשימה.

המחלקה Similarity:
```{r}
Class Similarity
WORD = re.compile(r'\w+')
```

מחלקה זו נועדה להוות מימוש של מדדי דימיון שונים בין מסמכים. זאת ע"מ להעריך את איכות דיוק החיזוי שלנו.

```{r}
def get_ngram(text1,text2):
    return NGram.compare(text1, text2)
```
פונקציה זו מחשבת את מדד הדימיון הידוע ngram similarity

```{r}
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator
```
פונקציה זו מחשבת את המדד  מקבלת שני וקטורים ומחשבת עבורם את המדד cosine similarity.

מדד זה מודד את הדמיון בין שני וקטורים  ע"י חישוב הקוסינוס של הזווית ביניהם.
משקולת זו הוא מדד של אורינטציה ולא של חשיבות, ניתן להשתמש בו כהשואה בין מסמכים על מרחב מנורמל מכיוון שאנחנו לא לוקחים בחשבק את החשיבות של כל המדד tf idf

של כל מילה אלא את הזווית בין המסמכים.

מדד זה ייצר לנו משקולת שתייצג כמה קשורים הם שני מסמכים ע"י התבוננות בזווית במקום בחשיבות.
```{r}
def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)
```
פונקציה זו ממירה מחרוזת לוקטור של מילים(מערך של מילים) ע"מ שנוכל לחשב על המסמכים את מדד הדמיון
```{r}
def get_text(path):
    content = ""
    for fname in glob.glob(path):
        with open(fname, 'r') as content_file:
            content = content + content_file.read()
    return content
```
פונקציה זו קוראת את כל הטקסט מנתיב מסוים שנשלח לה כקלט.

###מדד LSH:

מדד זה מפחית את את המימדיות של נתונים בעלי מימדיות גבוהה.
המדד מבצע האשינג בכך שהוא ממפה פריטים דומים לאותם "דליים" בהתסברות גבוהה.
כלומר, המדד מבצע השוואה בין כל משפט בטקסט הנחזה למשפטים בטקסט המקור ובודק לאיזה משפטים הוא הכי דומה.

להלן הפלט:

    4 -->  1307   0.20     0.09
      7 -->  1231   0.60     0.25
      7 -->  1370   0.30     0.07
      7 -->  1418   0.30     0.07
     19 -->    57   0.20     0.10
     19 -->   345   0.20     0.07
     19 -->  1258   0.20     0.10
     31 -->  1192   0.20     0.05
     31 -->  1193   0.20     0.06
     31 -->  1457   0.20     0.15
     31 -->  1526   0.20     0.15
     31 -->  1605   0.20     0.03
     31 -->  1735   0.20     0.03
     34 -->    98   0.20     0.03
     34 -->   900   0.20     0.03
     34 -->  1014   0.20     0.03
     34 -->  1367   0.20     0.02
     34 -->  1415   0.20     0.02
     34 -->  1619   0.20     0.03
     34 -->  1749   0.20     0.03
     59 -->     4   0.20     0.03
     59 -->   158   0.20     0.02
     63 -->   756   0.20     0.02
     63 -->   777   0.20     0.02
     64 -->   111   0.20     0.03
     64 -->  1193   0.20     0.04
     71 -->    98   0.20     0.06
     71 -->   900   0.20     0.05
     71 -->  1014   0.20     0.05
     71 -->  1367   0.20     0.03
     71 -->  1415   0.20     0.03
     71 -->  1619   0.20     0.05
     71 -->  1749   0.20     0.05
     73 -->  1136   0.20     0.04
     73 -->  1137   0.20     0.03
     73 -->  1139   0.20     0.05
     73 -->  1188   0.20     0.03
     73 -->  1206   0.20     0.04
     73 -->  1207   0.20     0.04
     73 -->  1277   0.20     0.05
     73 -->  1289   0.20     0.03
     73 -->  1295   0.20     0.01
     92 -->    98   0.20     0.10
     92 -->   900   0.20     0.08
     92 -->  1014   0.20     0.08
     92 -->  1367   0.20     0.05
     92 -->  1415   0.20     0.05
     92 -->  1619   0.20     0.08
     92 -->  1749   0.20     0.08
     97 -->   440   0.20     0.07
     97 -->   653   0.20     0.02
     97 -->  1087   0.20     0.03
    100 -->  1153   0.90     0.95
    
    העמודה השמאלית מייצגת את המספר הסידורי של המשפט החזוי.
    עמודה השנייה משמאל מייצגת את המספר הסידורי של המשפט בטקסט המקורי שהכי דומה למשפט חזוי זה.
    לאחר מכן, הערך של מדד הדימיו, כלומר כמה המשפט הזה דומה לו.
העמודה הכי ימנית מייצגת את  את היחס בין חיתוך המשפטים בין שני הקבצים לבין האיחוד של המשפטים בשני הקבצים.

לדוגמא:

  7 -->  1231   0.30     0.25   
  7 So the Fed did the difference we basis where.
    1332 So the Fed did it.
    
19 -->  1258   0.20     0.10   19 And this sides system is the same thing.
    1359 We're still talking about the same thing.
    
91 -->  1680   0.20     0.09   91 The whole deep system is set up roll.
    1781 The whole clearing system is set up t make sur that you, you almost never have to pa with reserves.
Chat Conversation End

ניתן לראות את הדמיון בין המשפטים החזויים לבין המשפטים המקוריים.

##שלב שמיני- ניתוח תוצאות המחקר והסקת מסקנות:
###מדד הדימיון cosine similarity:

מדד הדימיון קיבל את הערך 0.906 מתוך טווח ערכים אפשרי בין -1 ל1.
כידוע ככל שהערך של מדד דימיון זה יותר קרוב ל1 משמע הדברים שהזווית בין הוקטורים  קטנה יותר, כלומר שהמרחק ביניהם קטן יותר ועל כן משמע הדבר שהם קרובים יותר ודומים יותר.

משמעות ערך גבוה זה של מדד הדמיון הוא שהטקסט אשר הופק ע"י מודל החיזוי בעל דימיון גבוה לטקסט המקורי והמשפטים אשר חוזו ע"י המודל בעלי דמיון רב למשפטים אשר מריבים את הטקסט המקורי.

יש לציין שציון גבוה זה לא התקבל עבור המודל שלנו על ההתחלה. ושכן היה עלינו  להפיק לקחים, להסיק מסקנות ולבצע שיפורין ולאמן את המודל כל פעם בצורה טובה יותר ע"מ להגיע לתוצאות איכותיות אלו.

ע"מ לשפר את המודל ביצענו תהליך של ניסוי וטעייה פעמים רבות עד להגעה לתוצאה הטובה ביותר.
###מדד הדימיון ngram similarity:
מדד זה קיבל את הערך 0.82

###הצעדים שביצענו לשיפור דיוק המודל:

* הורדת סימני פיסוק- לאחר מספר ניסיונות הבחנו כי הסרת סימני הפיסוק מהטקסט המקורי עוזרים להגדיל את דיוק החיזוי. זאת מכיוון שהחיזוי לא חוזה סימני פיסוק אלא משפטים בלבד ולכן סימני הטקסט עלולים לפגום בלמידת המודל ולבלבל אותו עם סימנים לא רלוונטים. על כן, לאחר שהסרנו את סימני הפיסוק מהטקסט והלמידה בוצעה על המשפטים בלבד- החיזוי עבורם היה איכותי יותר וקיבל ציון גבוה יותר.
* ביצענו מיון של המילים בטקסט ע"פ שכיחות הופעתן. באמצעות מיון זה לקחנו רק את 2000 המילים השכיחות ביותר בטקסט אשר יהוו את הקורפוס שלנו ואשר הם יהיו המילים החזויות. הסיבה שביצענו זאת היא מכיוון שהבחנו שכאשר המודל מנסה ללמוד על מילים נדירות הוא לא מהווה עבורן חיזוי טוב מכיוון שאין לו מספיק מידע מקור ע"מ ללמוד עליהן. בנוסף לכך, הדבר שיפר ביצועים מבחינת זמן ריצה 
.את המילים הנדירות החלפנו בייצוג אחיד לכל המילים הלא שכיחות
דבר זה שיפר את התוצאות שכן מילים נדירות לא נכנסו למילון ואת כולן ייצג הערך 
UNKNOWN_TOKEN
דבר זה שיפר את חיזוי המודל מכיוון שבאופן טבעי מעצם היותן מילים נדירות לא היה לו מספיק נתונים ללמוד עליהם ולחזות אותן בצורה ראויה, לכן הסרתן שיפרה את התוצאות.
* ביצענו 50 הרצות של המודל שכן הבחנו שמספר הרצות זה גורם לכך שהקצב למידה מתייצב והטעות קטנה.
* הגדרנו כי אורך מינימלי של משפט שנחזה יהיה 7 מילים, מכיוון שחלק מחקירת אוסף הנתונים הבחנו שזהו האורך הממוצע של משפט בו
* בנוסף, הגדרנו כי החיזוי יחזה 100 משפטים- שכןלאחר מספר ניסיונות חיזוי עם מספר משפטים שונה הבחנו כי מספר זה נותן מדגם מספיק מייצג ומאפשר הבנה יותר רחבה של הטקסט ואופיו.
* מתן פרמטרים לגודל של השכבה החבויה
hiddenDimention=100
זאת ע"מ להרוויח את מרחב החיפוש בכל איטרציה. לאחר מס' בדיקות הבחנו כי מספר זה הוא היעיל והאידיאלי ביותר.


