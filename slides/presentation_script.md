# Presentation Script

**Thijesh:** Slides 1–9 · **Uday:** Slides 10–18 and the demo

---

## Before the talk

1. Open a terminal in `C:\codes\nlp-aat` and start the app:
   ```powershell
   .venv\Scripts\python.exe -m uvicorn app.api:app --port 8000
   ```
2. Wait for "Application startup complete".
3. Open this link and let it load once: http://127.0.0.1:8000/?sample=test_resume_082.txt&mode=compare&view=text

---

## Thijesh

**Slide 1: Title**
Good morning. I am Thijesh, and this is Uday. Our project is Resume NER: Pretrained versus Fine-Tuned BERT. We took a BERT model trained on news articles, tested it on resumes, then trained it further on resumes and measured how much it improved.

**Slide 2: Agenda**
I will explain the problem, the dataset and the model. Uday will show the results and a live demo, and then conclude.

**Slide 3: Introduction**
Resumes come as PDF or Word files, and every resume has a different layout. Companies need the important details pulled out automatically: name, email, college, degree, companies and skills. This task is called Named Entity Recognition, or NER. Our main result: accuracy went from 10% before training to 45% after training.

**Slide 4: What is NER**
NER means labelling each word with a type. In "Ashok Kunam, Team Lead, Microsoft", Ashok Kunam is a NAME, Team Lead is a DESIGNATION, and Microsoft is a COMPANY. We use 10 entity types, shown in the table. The original model was trained on news, so it only knows person, organisation and location. Seven of our ten types are new to it.

**Slide 5: Application**
We built a working system. You upload a resume, the model reads it, and it returns the details as structured data. It takes about one second per resume on a normal laptop CPU.

**Slide 6: Dataset**
We used 219 labelled resumes from Kaggle. We used 153 for training, 33 for checking during training, and 33 for the final test. Most words are not entities, and some labels in the dataset are wrong or missing. This makes the task harder.

**Slide 7: Preprocessing**
We cleaned the text, removed one duplicate, and fixed broken labels. Then we converted each label into a tag on every word. We did not lowercase the text, because capital letters help the model find names. Long resumes are read in overlapping chunks, because BERT can only read about 512 pieces of text at a time.

**Slide 8: Model**
Our model is BERT, with 12 layers and 108 million parameters. "Before" means using the news-trained model as it is. "After" means giving it a new output layer for our 10 types and training it on resumes. Both are tested on the same resumes, so the comparison is fair.

**Slide 9: Training**
We tried six combinations of learning rate and batch size. We picked the best one using the validation data only, not the test data. Training took about 3.5 minutes on a laptop GPU. Now Uday will show the results.

---

## Uday

**Slide 10: Results**
Thanks, Thijesh. The F1 score went from 10.3% to 44.8%, which is more than four times better. Even on the three types the old model already knew, it went from 12% to 57%. Names improved the most: from 25% to 83%.

**Slide 11: Cross-validation and speed**
To check that this was not luck, we repeated the experiment on five different splits. The improvement was the same every time, around 41% against 10%. The model is 411 MB and takes about one second per resume.

**Slide 12: Examples**
Here is the same text read by both models. The old model calls a university a company and misses job titles. The new model finds the name, title, company, degree, college and year correctly.

**Slide 13: Ablations and errors**
We changed one setting at a time to see what matters. Training for fewer epochs, or cutting long resumes short, made the results worse. Most remaining errors are partly wrong boundaries or extra guesses. Skills are the hardest type, because the dataset labels whole paragraphs as one skill.

**Slide 14: Live demo**
Now let me show the working app. *(Do the demo steps below.)*

**Slide 15: Discussion**
Fine-tuning helps because resumes need new labels and new rules. BERT already understands English, so a small dataset was enough to give a big improvement. Our main problem was the dataset: some labels were missing or wrong.

**Slide 16: Conclusion**
Fine-tuning improved F1 from 10% to 45%. A pretrained model does not work well on a new kind of text with new labels, but a small amount of training fixes much of it. In future, we would use better data, models that understand page layout, and check the system for bias before it is used in hiring.

**Slide 17: References**
These are the papers, the model and the dataset we used.

**Slide 18: Thank you**
Thank you. We are happy to take questions.

---

## Demo steps (Uday)

1. **Show the comparison page** (it is already open).
   Say: "This is a test resume the model has never seen. On the left is the old model, and on the right is our trained model. Each colour is a type of entity. The right side finds far more, and far more correctly."
2. **Click the Fields view.**
   Say: "Here are the details pulled out as clean fields: name, email, skills, education and companies."
3. **Upload `data\sample_resumes\sample_fictional.pdf`.**
   Say: "It works on real PDF files too, and it takes about one second."
4. **Click the Results tab.**
   Say: "These are the same results from our slides."
5. **Go back to the slides.**

If the demo fails, show the screenshots on Slide 14 and explain them.

---

## If someone asks

- **Why is 45% good?** We count an entity as correct only if every word matches exactly. Also, it is four times better than before.
- **Why not use the best test score?** Choosing settings by test score would be cheating. We chose using the validation data only.
- **How are long resumes handled?** In overlapping chunks of 512 tokens.
- **Why are skills so low?** The dataset labels whole skill paragraphs as one entity, so exact matches are rare.
