from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

actual = ['m', 'nm', 'm', 'm', 'm', 'nm', 'nm', 'm', 'm', 'nm', 'nm', 'm', 'nm', 'm', 'nm', 'nm', 'nm', 'm', 'm', 'nm']
predicted = ['m', 'nm', 'm', 'm', 'm', 'nm', 'nm', 'm', 'm', 'nm', 'nm', 'm', 'nm', 'm',
             'nm', 'nm', 'nm', 'nm', 'm', 'nm']

labels = ['m', 'nm']
results = confusion_matrix(actual, predicted)

print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(actual, predicted))
print('Classification Report : ')
print(classification_report(actual, predicted))

ax = plt.subplot()
sns.heatmap(results, annot=True, ax=ax)

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Cancerous (65% THP)', 'Non-cancerous (65% PHA)'])
ax.yaxis.set_ticklabels(['Non-cancerous (65% PHA) ', 'Cancerous (65% THP)'])
plt.savefig('confusion_matrix.png')
plt.show()
