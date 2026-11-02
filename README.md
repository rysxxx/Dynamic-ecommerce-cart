# Dynamic-ecommerce-cart

## About the Project

Dynamic E-Commerce Cart is a Python-based shopping application built using Streamlit. It simulates a basic online shopping experience where users can browse products, search and filter items, manage their shopping cart, apply discount coupons, and generate bills.

The project also includes a **Machine Learning-based product recommendation system** that suggests similar products based on their categories and tags. It uses CountVectorizer and Cosine Similarity to identify products with similar features.

The application also supports automatic discounts, coupon codes, GST calculation, stock management, and order history, making the shopping experience more interactive and realistic.

## Key Features

- Browse, search, and filter products.
- Add, remove, and update cart items.
- Get product recommendations using Machine Learning.
- Apply discount coupons and automatic discounts.
- Calculate GST and generate detailed bills.
- Manage product stock availability.
- View placed orders.

## Technologies Used

- **Python** – Core programming language.
- **Streamlit** – Web application interface.
- **Pandas** – Data handling and processing.
- **Scikit-learn** – Product recommendation system.
- **CountVectorizer** – Text feature extraction.
- **Cosine Similarity** – Finding similar products.
- **CSS** – Custom interface styling.

## Machine Learning Approach

The project uses **Content-Based Filtering** to recommend products. It analyses product categories and tags, converts them into numerical vectors using CountVectorizer, and calculates similarities using Cosine Similarity. Based on these scores, the application suggests products similar to those in the user's cart.

## Installation and Execution

1. Clone the repository:

   ```bash
   git clone https://github.com/rysxxx/Dynamic-ecommerce-cart.git
   ```

2. Navigate to the project folder:

   ```bash
   cd Dynamic-ecommerce-cart
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Future Improvements

- User login and authentication.
- Online payment integration.
- Database integration.
- Product ratings and reviews.
- More advanced recommendation algorithms.

---

**Subject:** Python for Machine Learning (ML)  
**Project:** Dynamic E-Commerce Cart with Machine Learning-Based Recommendations
