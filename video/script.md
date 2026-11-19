# Machine Learning Models, explained — YouTube tutorial + one Reel
Narrator: VCTK p270. Every number is from notebooks/ml_models_live.ipynb (live run, 2026-11-19).
Transformer/LLM numbers are from the ScamGuard project's live run.

[Y01]
There are hundreds of machine learning models. But you only need to understand about fifteen to pick the right one for almost any problem. Today, each one runs on real data, in front of you.

[Y02]
First, the big idea. Every model is a function. Data goes in, a prediction comes out. Inside it, there are numbers called parameters. Training means adjusting those numbers, again and again, until the prediction error is as small as possible.

[Y03]
And the right model depends on your data. Numbers in a table. Text. Images. Things that change over time. Or data with no labels at all. Keep that map in mind. We will place every model on it.

[Y04]
Model one: linear regression. It predicts a number. Imagine guessing a house price from the owner's income. More income, higher price. Linear regression draws the straight line that fits those points best.

[Y05]
How does it find that line? It starts with a random line, measures the error, and moves the line a little in the direction that reduces the error. That is gradient descent. On real California data, sixty small steps took the error from five point six to zero point seven.

[Y06]
With all eight features, on twenty thousand six hundred and forty districts, it explains fifty eight percent of the price variation, and misses by about fifty three thousand dollars on average. It is your first baseline for any number. It is fast and fully explainable, but it only sees straight lines.

[Y07]
Model two: logistic regression. Same weighted sum, but for yes or no questions. Is this tumour benign? Is this email spam?

[Y08]
It squeezes the weighted sum through a curve shaped like the letter S, called the sigmoid, which turns any number into a probability between zero and one. Above zero point five, yes. Below, no.

[Y09]
On five hundred and sixty nine real tumours with thirty measurements, it was right ninety eight point six percent of the time. Use it as the first model for any yes or no problem on tables. Simple, fast, and its probabilities mean something.

[Y10]
Model three: k nearest neighbours. Tell me who your neighbours are, and I will tell you who you are. To classify a new point, it looks at the k closest examples, and lets them vote.

[Y11]
There is no training at all. It just remembers the data. Here are real wines, measured by alcohol and flavanoids. Every colour region shows where the neighbours vote for one grape type. With one neighbour, eighty five percent correct. With the best k, ninety one percent.

[Y12]
Use it for small datasets where similar things really behave the same. Avoid it for big data, because every prediction compares against every stored example.

[Y13]
Model four: the decision tree. It works like a guessing game with yes or no questions. Is the tumour radius above sixteen point eight? Yes or no. Then the next question, and the next.

[Y14]
At every step, it picks the question that splits the data into the purest groups. This real tree has only three levels, and you can read every rule. That is its superpower: you can explain each decision to a doctor.

[Y15]
But a single deep tree memorises its training data. On the test set, one full tree scored ninety two point three percent.

[Y16]
Model five: random forest. Instead of asking one expert, ask three hundred, and take the vote. Each tree sees a random sample of rows and a random subset of features, so their mistakes are different.

[Y17]
Different mistakes cancel out. Three hundred trees raised the accuracy from ninety two point three to ninety five point eight percent. Use it as a strong, safe default for tabular data.

[Y18]
Model six: gradient boosting. Think of a student who studies only yesterday's mistakes. The first tree makes a rough guess. The second tree learns only the errors of the first. The third fixes what is still wrong.

[Y19]
On the house prices, the error fell from one point one on the first tree to zero point four four after six hundred trees. Linear regression explained fifty eight percent. Random forest, eighty one percent. LightGBM, eighty five percent.

[Y20]
For tables of numbers and categories, gradient boosting, like LightGBM or XGBoost, is usually the strongest choice. It wins most tabular competitions.

[Y21]
Model seven: the support vector machine. Imagine building a road between two villages. A good road keeps the widest possible distance from the houses on both sides.

[Y22]
That is exactly what it does: it finds the boundary with the widest margin. Only the points closest to the road decide where it goes. Here, twenty eight support vectors define the whole boundary.

[Y23]
With a kernel, it can bend the road into curves. Using all thirteen features of each wine, a curved kernel was right ninety eight percent of the time. Use it for small or medium data with many features.

[Y24]
Model eight: naive Bayes, the classic for text. If a message says tires, brake and dealer, it is probably about cars.

[Y25]
It counts how often each word appears in each topic, and multiplies those probabilities with Bayes' rule. It naively assumes words are independent. That is wrong, but it works surprisingly well.

[Y26]
On two thousand three hundred and sixty five real forum posts, across four topics, it was right eighty eight percent of the time, in a fraction of a second. Use it as a fast first baseline for text.

[Y27]
Model nine: k means. Now there are no labels. Imagine sorting a crowd into groups without knowing anyone's name.

[Y28]
Pick k centres. Assign every point to its nearest centre. Move each centre to the middle of its group. Repeat until nothing changes. On real wines, it converged in six steps.

[Y29]
It never saw the wine labels, yet its three groups matched them with an adjusted Rand index of zero point nine two. Use it to find segments, like customer groups, when nobody has labelled the data.

[Y30]
Model ten: principal component analysis. Think of the shadow of a three dimensional object. A good angle keeps most of the shape in just two dimensions.

[Y31]
Principal component analysis finds the directions where the data varies the most. Each digit image has sixty four pixels. Just two directions keep twenty eight percent of the information, and twenty one keep ninety percent.

[Y32]
Use it to visualise data, to compress features, and to remove noise before another model.

[Y33]
Model eleven: the neural network. Each neuron computes a weighted sum, then a simple function decides how strongly it fires. Stack them in layers, and the network learns its own features.

[Y34]
Training is backpropagation: measure the error at the output, and send it backwards to adjust every weight. This network has four thousand eight hundred and ten weights. Its loss fell from two point three to almost zero.

[Y35]
On the digit test set, it was right ninety eight point two percent of the time. Neural networks shine when there is a lot of data and the patterns are complex.

[Y36]
Model twelve: the convolutional neural network, made for images. Imagine a small magnifying glass sliding across a photo, looking for edges, then curves, then whole shapes.

[Y37]
Each filter is a small grid of learned weights. These are the real eight filters the network learned in its first layer, and here is what they see in an image of the number seven.

[Y38]
Trained for two epochs on sixty thousand images, with fifty four thousand weights, it read ninety seven point five percent of ten thousand new digits correctly. For images, start with a CNN.

[Y39]
Model thirteen: the LSTM, for sequences. It reads data in order and keeps a memory. Small gates inside it decide what to remember, what to forget, and what to pass on.

[Y40]
I gave it forty three years of carbon dioxide levels from Mauna Loa, and asked it to forecast the last three years. And here is an honest lesson. A simple rule, last year's pattern plus last year's rise, missed by one ppm. The L S T M model missed by three.

[Y41]
On five hundred and twenty six data points, the simple baseline won. Deep learning needs lots of data. Always compare against a simple baseline before you trust a complex model.

[Y42]
Model fourteen: the Transformer, the engine behind modern artificial intelligence. Every word looks at every other word at the same time. That is attention.

[Y43]
In my ScamGuard project, a fine tuned Transformer split a Bangla scam message into twenty six tokens. Its attention focused on bKash, PIN and verify, and it was ninety nine point nine percent sure it was a scam.

[Y44]
Model fifteen: the large language model. At its core, it predicts the next word. Asked whether a message was a scam, the next word Yes had a probability of zero point nine nine nine eight.

[Y45]
On a locked real world test, that model, which was never trained on the data, scored zero point nine six, the best of five models. Use Transformers and large language models for language tasks, when you have the compute.

[Y46]
So which model should you use? Numbers in a table: start with linear or logistic regression, then gradient boosting. Text: naive Bayes as a baseline, then a Transformer. Images: a CNN. Sequences: a simple baseline first, then an LSTM or a Transformer. No labels: k means and principal component analysis.

[Y47]
And always: divide your data honestly, compare against a baseline, and trust only the test set you never touched. That is machine learning, one model at a time. The code and every result are on my GitHub.

# Reel (one standalone Reel, under 3 minutes)

[R01]
Fifteen machine learning models. Which one should you use? Here is how each one actually decides, on real data.

[R02]
Every model is a function with adjustable numbers. Training means nudging those numbers so the error gets smaller. That is gradient descent.

[R03]
Linear regression draws the best straight line for a number. Logistic regression bends it through a sigmoid into a probability. With real tumour data, it was ninety eight point six percent correct.

[R04]
A decision tree asks yes or no questions. A random forest asks three hundred trees and takes the vote: ninety two point three became ninety five point eight percent.

[R05]
Gradient boosting builds each tree to fix the last one's mistakes. On house prices it beat linear regression, fifty eight to eighty five percent. For tables, it is usually the winner.

[R06]
No labels? K means moves centres until the groups stop changing. On wines it never saw labelled, it matched the real grape types closely: zero point nine two.

[R07]
Images? A CNN slides small filters across the picture, filters it learned by itself. These are its real filters. It read ninety seven point five percent of new digits correctly.

[R08]
Sequences? An LSTM keeps a memory. But on forty three years of carbon dioxide, a one line baseline beat it. Always test a baseline first.

[R09]
Language? A Transformer lets every word attend to every other word. A large language model predicts the next word: Yes, zero point nine nine nine eight.

[R10]
Tables: gradient boosting. Text: Transformers. Images: CNNs. No labels: k means. The full tutorial, with every model in depth, is on YouTube. Which one do you use? Comment below.
