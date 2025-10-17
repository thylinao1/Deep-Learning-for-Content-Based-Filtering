🎬 DeepMovieMatch

DeepMovieMatch is a neural network–based movie recommender system that predicts user preferences using learned embeddings rather than simple genre matching. It builds a shared latent space where users and movies are represented as vectors, and their similarity determines how much a user will like a movie.

🚀 Overview

Unlike traditional content-based or collaborative filtering systems that rely on manually defined features or linear similarity, DeepMovieMatch uses deep learning to learn complex, nonlinear relationships between users and movies.
The project implements a two-tower neural architecture:
User Tower: Encodes user attributes such as average ratings and genre tendencies.
Movie Tower: Encodes movie characteristics such as genre and popularity.

The two embeddings are combined via a dot product to predict ratings or preference scores.

🧠 Key Features
1. Learns latent representations for users and movies automatically.
2. Models nonlinear feature interactions through multi-layer perceptrons.
3. Produces a latent embedding space where similar movies cluster together.
4. Supports user-based and movie-based recommendations.
5. End-to-end training with TensorFlow/Keras

Why this is cool (vs. simple vector matching)?

What’s really cool about this whole thing is that the model isn’t just comparing rows of 1s and 0s to see which genres overlap. 
It’s actually learning what kinds of hidden relationships exist between users and movies. 
Instead of you deciding what “Action + Sci-Fi” means, the neural network figures that out on its own. 
The dense layers take all those simple input features and turn them into something much deeper, continuous embeddings that capture subtle patterns in user preferences and movie characteristics.

The result of this is a latent space, a kind of hidden map where similar users and movies end up close to each other, even if their raw genre flags look completely different. 
Two movies might have no genres in common, but if the same types of people love them, the network will place them near each other in that space. 
That’s something you’d never get from just comparing Boolean vectors.

And because it’s trained end-to-end, the whole model learns automatically what matters most for predicting whether a user will like a movie. 
It’s not just about matching surface-level features anymore, it’s about discovering a deep, meaningful structure underneath the data. 
That’s what makes this approach so much more powerful (and honestly, way more exciting) than simple vector matching.



The recsysNN_utils.py module imported ### Utilities
- `load_data()` – loads matrices, feature names, and metadata.
- `pprint_train(...)` – quick visual sanity check for training slices.
- `gen_user_vecs(...)` – repeats one user row across all items for batch scoring.
- `print_pred_movies(...)` – shows top-N predicted titles.
- `get_user_vecs(...)` – constructs (user,item) pairs and labels for evaluation.
- `print_existing_user(...)` – side-by-side predicted vs. actual for a user.
