# Ad Recommendation System
## An Experiment in Understanding Ad Recommendation/Curation Systems

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Next.js](https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

An advanced ad recommendation system leveraging machine learning to simulate targeted advertising on platforms like Instagram and YouTube.

[Genesis](#genesis) • [Features](#features) • [Tech Stack](#tech-stack) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [API Reference](#api-reference) • [Contributing](#contributing)

</div>

## Features

- 🧠 Two Tower Neural Network for efficient user-ad matching
- 🔄 A/B Testing support for model comparison
- ⚡ Real-time inference using FastAPI
- 📊 Interactive data visualization with Chart.js
- 🎨 Sleek, responsive UI built with Next.js and Tailwind CSS

## Tech Stack

### Backend
- [FastAPI](https://fastapi.tiangolo.com/): High-performance web framework for building APIs
- [TensorFlow](https://www.tensorflow.org/): Open-source machine learning platform
- [Pydantic](https://pydantic-docs.helpmanual.io/): Data validation using Python type annotations
- [Redis](https://redis.io/): In-memory data structure store for caching and session management

### Frontend
- [Next.js](https://nextjs.org/): React framework for server-side rendering and static site generation
- [TypeScript](https://www.typescriptlang.org/): Typed superset of JavaScript
- [Tailwind CSS](https://tailwindcss.com/): Utility-first CSS framework
- [Shadcn UI](https://ui.shadcn.com/): Re-usable components built with Radix UI and Tailwind
- [React Hook Form](https://react-hook-form.com/): Performant, flexible and extensible forms
- [Chart.js](https://www.chartjs.org/): Simple yet flexible JavaScript charting library

## Architecture

### The Genius of Dual Processing
The Two Tower architecture, as its name suggests, consists of two parallel neural networks:

1. **User Tower**: This network processes user features such as demographics, browsing history, and interaction patterns. It learns to create a dense representation (embedding) of each user in a high-dimensional space.

2. **Ad Tower**: Similarly, this network processes ad features like content category, target demographics, and historical performance. It creates embeddings for ads in the same high-dimensional space as the users.

The brilliance of this approach lies in its ability to process user and ad data independently, allowing for efficient pre-computation and caching of ad embeddings.

### The Magic Moment: Similarity Computation
Once we have our user and ad embeddings, the magic happens in the similarity computation step. This is typically a simple operation like dot product or cosine similarity, which can be computed extremely quickly even for large numbers of ads.

#### Why It Works So Well
1. **Scalability**: By pre-computing ad embeddings, we can handle millions of ads and users efficiently.
2. **Flexibility**: Each tower can be independently updated or modified, allowing for easy incorporation of new features or architectural improvements.
3. **Cold Start Handling**: New users or ads can be immediately integrated into the system based on their features, without needing historical interaction data.

This architecture forms the backbone of recommendation systems used by tech giants like YouTube and Instagram, enabling them to serve personalized content to billions of users in real-time.

### System Architecture: Bringing It All Together

Our system architecture is designed to handle the entire lifecycle of ad recommendations, from data ingestion to user interaction. Here's a high-level overview:

#### Key Advantages:
- **Scalability**: Can handle large numbers of users and items efficiently
- **Flexibility**: Allows for incorporating various types of features
- **Fast Retrieval**: Enables quick similarity computations in high-dimensional spaces

#### Process Flow:
1. User and ad features are preprocessed and fed into their respective towers.
2. Each tower produces a dense embedding vector (e.g., 16-dimensional).
3. Similarity between user and ad embeddings is computed (e.g., dot product).
4. Ads are ranked based on their similarity scores with the user.

### System Architecture

1. **Data Ingestion**: User profiles and ad information are collected and preprocessed.
2. **Feature Engineering**: Raw data is transformed into meaningful features for the neural networks.
3. **Model Training**: Two Tower Neural Network is trained on historical user-ad interaction data.
4. **Inference Service**: FastAPI server hosts the trained model for real-time predictions.
5. **A/B Testing**: Multiple model variants are deployed and compared.
6. **Frontend**: Next.js application provides an interactive interface for users and displays recommendations.

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- Redis

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ad-recommendation-system.git
cd ad-recommendation-system/backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn inference_service:app --reload
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

## Usage

1. Access the frontend application at `http://localhost:3000`.
2. Input user characteristics in the provided form.
3. View generated ad recommendations and their scores.
4. Explore different user profiles and ad combinations to see varying results.

## API Reference

### Recommend Ads
```http
POST /recommend
```
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `user` | `object` | User features including demographics and interests |
| `ads` | `array` | List of ad objects with their characteristics |

#### Response
```javascript
{
  "ad_scores": [float],
  "explanation": [string],
  "variant": string
}
```

For more details, refer to the [API Documentation](docs/api.md).

## Contributing

This project is still super new and just shows a single implementation method and the front-end part of it definitely needs a lot of work. In that effort, especially since I'm not really an expert at either, contributions are most certainly welcome! Please feel free to submit a Pull Request:)

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Recommenders](https://www.tensorflow.org/recommenders)
- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
