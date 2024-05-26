# Recommendation-System-Streamers-on-Twitch
This is a Hybrid Recommendation model that recommends streamers on Twitch. This is a project for Unsupervised Machine Learning Fall 2022.

Data Collection: scraped Twitch API on Nov. 14, 2022 using Selenium to extract game and genre information, streamers and the games they play, streamers’ account information, and users’ following data (which streamers users follow). The top 100 streamers were chosen, and then roughly 15000 users’ following information around these streamers was collected.

The models were trained using a 75-25 train-test split.

Content-based Filtering: Game descriptions were cleaned using ntlk and vectorized used TF-IDF. Cosine similarity was used to compute similar games.

Item-item Collaborative Filtering: Users' following status of the top 100 streamers was organized into a ratings matrix, where the users are followers and items are the streamers. Unary ratings (follow or not-follow) allowed for Jaccard similairty to be used to compute 3 closest streamers to a selected streamer based on user preferences.

Hybrid Model: The hybrid model combined the content-based and collaboritive filtering models in an unweighted format.

Recall was used to evaluate the 3 models. The content-based model did not perform well on its own, whereas the collaborative filtering model made better recommendations. The hybrid model further improves the collaborative filtering model by using information on both the streamers and watchers.
<img width="975" alt="Screenshot 2024-05-26 at 5 37 24 PM" src="https://github.com/slakhiani/Recommendation-System-Streamers-on-Twitch/assets/135447183/10b37db8-de16-44fd-a035-5c64a55f72d3">
