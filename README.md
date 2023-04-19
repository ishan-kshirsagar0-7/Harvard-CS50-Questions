# Harvard CS50 AI (Week 6) Project 12 : Parser

This is the final project of [Harvard CS50's Introduction to Artificial Intelligence course](https://cs50.harvard.edu/ai/2020/), the second project of Week 6.

## My Outputs

![Screenshot of my output on terminal](https://cdn.discordapp.com/attachments/1091358303063396496/1098339817655373926/image.png)

## Objective

Write an AI to answer questions.

## Background

Question Answering (QA) is a field within natural language processing focused on designing systems that can answer questions. Among the more famous question answering systems is Watson, the IBM computer that competed (and won) on Jeopardy!. A question answering system of Watson’s accuracy requires enormous complexity and vast amounts of data, but in this problem, we’ll design a very simple question answering system based on inverse document frequency.

Our question answering system will perform two tasks: document retrieval and passage retrieval. Our system will have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. Once the top documents are found, the top document(s) will be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined.

How do we find the most relevant documents and passages? To find the most relevant documents, we’ll use tf-idf to rank documents based both on term frequency for words in the query as well as inverse document frequency for words in the query. Once we’ve found the most relevant documents, there many possible metrics for scoring passages, but we’ll use a combination of inverse document frequency and a query term density measure.
