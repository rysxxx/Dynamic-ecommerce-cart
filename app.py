# ------------------------------------------------------------
# Dynamic E-Commerce Cart  |  Python for Machine Learning Project
# Tools: Streamlit (UI), Pandas (data), Scikit-learn (recommendations)
# Run with:  streamlit run app.py
# ------------------------------------------------------------
import os
import streamlit as st
from PIL import Image, ImageOps
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- SETTINGS (easy to change) ----------
GST_RATE = 0.18                 # 18% GST
AUTO_DISCOUNT_LIMIT = 2000      # spend this much -> automatic discount
AUTO_DISCOUNT_PERCENT = 10      # 10% automatic discount
COUPONS = {                     # coupon code : (type, value, minimum cart value)
    "SAVE10": ("percent", 10, 500),
    "FLAT200": ("flat", 200, 1000),
    "WELCOME5": ("percent", 5, 0),
}

st.set_page_config(page_title="Dynamic E-Commerce Cart", page_icon="🛒", layout="wide")


# ---------- STEP 1: LOAD DATA ----------
@st.cache_data
def load_products():
    return pd.read_csv("products.csv")

products = load_products()


# ---------- STEP 2: MACHINE LEARNING (recommendation engine) ----------
# Idea: turn each product's category + tags into numbers (bag of words),
# then use cosine similarity to find which products are most alike.
@st.cache_data
def build_similarity_matrix(df):
    text = (df["category"] + " " + df["tags"]).tolist()
    vectors = CountVectorizer().fit_transform(text)
    return cosine_similarity(vectors)

similarity = build_similarity_matrix(products)


def recommend(cart_ids, top_n=3):
    """Return up to top_n products similar to what is in the cart."""
    if not cart_ids:
        return []
    rows = [products.index[products["id"] == pid][0] for pid in cart_ids]
    scores = similarity[rows].sum(axis=0)      # add similarity of all cart items
    for r in rows:
        scores[r] = -1                         # do not recommend items already in cart
    best = scores.argsort()[::-1][:top_n]      # highest scores first
    return [products.iloc[i] for i in best if scores[i] > 0]


# ---------- STEP 3: SESSION STATE (remembers data while app runs) ----------
if "cart" not in st.session_state:
    st.session_state.cart = {}                                   # {product_id: quantity}
if "stock" not in st.session_state:
    st.session_state.stock = dict(zip(products["id"], products["stock"]))
if "coupon" not in st.session_state:
    st.session_state.coupon = ""
if "orders" not in st.session_state:
    st.session_state.orders = 0
if "order_history" not in st.session_state:
    st.session_state.order_history = []        # list of all placed orders
if "just_ordered" not in st.session_state:
    st.session_state.just_ordered = False


# ---------- STEP 4: CART FUNCTIONS ----------
def get_product(pid):
    return products[products["id"] == pid].iloc[0]


APP_DIR = os.path.dirname(os.path.abspath(__file__))


def find_image(p):
    """Look for the product picture. A .jpg/.jpeg/.webp with the same name
    overrides the built-in .png illustration, so you only need to drop a photo
    into the images folder with the same file name."""
    base = os.path.splitext(os.path.join(APP_DIR, p["image"]))[0]
    for ext in (".jpg", ".jpeg", ".webp", ".png"):
        if os.path.exists(base + ext):
            return base + ext
    return None


@st.cache_data
def load_square(path, size=400):
    """Centre-crop and resize ANY image to the same square size."""
    return ImageOps.fit(Image.open(path).convert("RGB"), (size, size))


def show_image(p, width=None):
    """Show the product picture (always the same size)."""
    path = find_image(p)
    if path is None:
        st.caption("(image missing)")
    elif width:
        st.image(load_square(path), width=width)
    else:
        st.image(load_square(path))


def add_to_cart(pid):
    current = st.session_state.cart.get(pid, 0)
    if current < st.session_state.stock[pid]:
        st.session_state.cart[pid] = current + 1
    else:
        st.toast("Sorry, no more stock available for this item!", icon="⚠️")


def decrease_qty(pid):
    st.session_state.cart[pid] -= 1
    if st.session_state.cart[pid] <= 0:
        del st.session_state.cart[pid]


def remove_item(pid):
    st.session_state.cart.pop(pid, None)


def clear_cart():
    st.session_state.cart = {}
    st.session_state.coupon = ""


def calculate_bill():
    """Return a dictionary with all bill amounts."""
    subtotal = sum(get_product(pid)["price"] * qty for pid, qty in st.session_state.cart.items())

    # automatic discount
    auto_discount = subtotal * AUTO_DISCOUNT_PERCENT / 100 if subtotal >= AUTO_DISCOUNT_LIMIT else 0

    # coupon discount
    coupon_discount = 0
    code = st.session_state.coupon
    if code in COUPONS:
        kind, value, minimum = COUPONS[code]
        if subtotal >= minimum and subtotal > 0:
            coupon_discount = subtotal * value / 100 if kind == "percent" else value

    discount = min(auto_discount + coupon_discount, subtotal)
    taxable = subtotal - discount
    gst = taxable * GST_RATE
    return {
        "subtotal": subtotal,
        "auto_discount": auto_discount,
        "coupon_discount": coupon_discount,
        "gst": gst,
        "total": taxable + gst,
    }


def make_receipt(bill):
    lines = ["===== DYNAMIC E-COMMERCE CART - RECEIPT =====",
             datetime.now().strftime("Date: %d-%m-%Y %H:%M"), ""]
    for pid, qty in st.session_state.cart.items():
        p = get_product(pid)
        lines.append(f"{p['name']:<25} {qty} x Rs.{p['price']} = Rs.{p['price'] * qty}")
    lines += ["", f"Subtotal        : Rs.{bill['subtotal']:.2f}",
              f"Auto Discount   : -Rs.{bill['auto_discount']:.2f}",
              f"Coupon Discount : -Rs.{bill['coupon_discount']:.2f}",
              f"GST (18%)       : Rs.{bill['gst']:.2f}",
              f"TOTAL           : Rs.{bill['total']:.2f}",
              "=============================================="]
    return "\n".join(lines)


# ---------- STEP 5: USER INTERFACE ----------
# ----- Custom look: background colour, banner, cards (CSS) -----
GROUP_NAME = "Monalisa, Tiyasa, Sumana, Monisha"   # <-- group member names

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500;600&display=swap');

/* extremely dark background */
.stApp {
    background: radial-gradient(circle at 15% 0%, #1a0a17 0%, #07040a 45%, #030204 100%);
    color: #DDFFF7;
}
/* fonts: Poppins for text, Playfair Display for headings */
html, body, .stApp, p, label, li, button, input, textarea,
[data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {
    font-family: 'Poppins', sans-serif !important;
}
h1, h2, h3, h4, [data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    color: #DDFFF7;
    letter-spacing: 0.3px;
}
.stApp p, .stApp label, .stApp li, .stApp span { color: #DDFFF7; }
[data-testid="stMetricValue"] { color: #FFA69E !important; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0511 0%, #14081a 100%);
    border-right: 1px solid rgba(147, 225, 216, 0.18);
}
/* inputs */
.stApp input, .stApp textarea {
    background: #1a0d20 !important; color: #DDFFF7 !important; border-radius: 10px;
}
div[data-baseweb="select"] > div {
    background: #1a0d20 !important; color: #DDFFF7 !important;
    border-color: rgba(147, 225, 216, 0.3) !important;
}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li {
    background: #1a0d20 !important; color: #DDFFF7 !important;
}
/* product cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, #170b1d, #0f0714);
    border: 1px solid rgba(147, 225, 216, 0.2) !important;
    border-radius: 18px;
    box-shadow: 0 6px 22px rgba(134, 22, 87, 0.35);
}
div[data-testid="stImage"] img { border-radius: 14px; }
.price { color: #93E1D8 !important; font-weight: 600; font-size: 1.1rem; }

/* buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid rgba(255, 166, 158, 0.35);
    background: linear-gradient(90deg, #861657, #AA4465);
    font-weight: 500;
}
.stButton > button p, .stDownloadButton > button p { color: #DDFFF7 !important; }
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(90deg, #AA4465, #FFA69E);
    border-color: #FFA69E;
}
.stButton > button:hover p, .stDownloadButton > button:hover p { color: #1a0612 !important; }

/* small buttons (−  +  Remove  ✕) */
.stButton > button { padding: 0.25rem 0.5rem; white-space: nowrap; min-height: 2.4rem; }
.qty { text-align: center; font-weight: 600; font-size: 1.2rem; padding-top: 6px; color: #DDFFF7; }

/* keep the top-right menu (Settings -> dark/light theme), hide only the Deploy button */
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
[data-testid="stDeployButton"] { display: none; }

/* tabs and order history boxes */
button[data-baseweb="tab"] p { font-size: 1.05rem; }
div[data-testid="stExpander"] {
    background: #140a1a;
    border: 1px solid rgba(147, 225, 216, 0.2);
    border-radius: 14px;
}

/* top banner */
.banner {
    background: linear-gradient(120deg, #861657 0%, #AA4465 70%, #d9706f 100%);
    padding: 30px 36px;
    border-radius: 22px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 166, 158, 0.35);
    box-shadow: 0 10px 30px rgba(134, 22, 87, 0.45);
}
.banner h1 { color: #DDFFF7 !important; margin: 0; font-size: 2.5rem; }
.banner p  { color: #DDFFF7 !important; margin: 6px 0 14px 0; font-size: 1.05rem; font-weight: 300; }
.badge {
    display: inline-block;
    background: rgba(7, 4, 10, 0.45);
    color: #DDFFF7 !important;
    border: 1px solid rgba(147, 225, 216, 0.5);
    padding: 5px 14px;
    border-radius: 20px;
    margin: 0 8px 6px 0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="banner">
    <h1>🛒 Dynamic E-Commerce Cart</h1>
    <p>Smart shopping with live discounts and AI-powered recommendations</p>
    <span class="badge">🎁 10% OFF above ₹{AUTO_DISCOUNT_LIMIT}</span>
    <span class="badge">🏷️ Use code SAVE10</span>
    <span class="badge">🤖 ML Recommendations</span>
    <span class="badge">👥 A project made by {GROUP_NAME} | Python for ML Project</span>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔍 Filter Products")
MAX_PRICE = int(-(-products["price"].max() // 50) * 50)   # highest price rounded up to a multiple of 50


def clear_filters():
    """Reset search box, category and price slider."""
    st.session_state.search = ""
    st.session_state.category = "All"
    st.session_state.max_price = MAX_PRICE


def clear_search():
    """Empty only the search box."""
    st.session_state.search = ""


st.sidebar.markdown("Search by name")
s1, s2 = st.sidebar.columns([5, 1.3])
search = s1.text_input("Search by name", key="search", placeholder="Type to search...",
                       label_visibility="collapsed")
s2.button("✕", key="clear_search_btn", on_click=clear_search, help="Clear search",
          use_container_width=True)
categories = ["All"] + sorted(products["category"].unique().tolist())
category = st.sidebar.selectbox("Category", categories, key="category")
max_price = st.sidebar.slider("Max price (₹)", 100, MAX_PRICE, MAX_PRICE, step=50, key="max_price")
st.sidebar.button("🧹 Clear filters", on_click=clear_filters, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    f"🎁 **Offers**\n\n- {AUTO_DISCOUNT_PERCENT}% off on orders above ₹{AUTO_DISCOUNT_LIMIT}\n"
    "- Coupons: `SAVE10`, `FLAT200`, `WELCOME5`"
)

# Apply filters
shown = products.copy()
if search:
    shown = shown[shown["name"].str.contains(search, case=False)]
if category != "All":
    shown = shown[shown["category"] == category]
shown = shown[shown["price"] <= max_price]

tab_shop, tab_orders = st.tabs(["🛍️ Shop", "📦 My Orders"])

with tab_shop:
    left, right = st.columns([3, 2], gap="large")

    # ----- LEFT: product catalog -----
    with left:
        st.subheader("🛍️ Products")
        st.caption(f"Showing {len(shown)} of {len(products)} products")
        if shown.empty:
            st.warning("No products match your filter.")
        rows = [shown.iloc[i:i + 3] for i in range(0, len(shown), 3)]
        for row in rows:
            cols = st.columns(3)
            for col, (_, p) in zip(cols, row.iterrows()):
                with col:
                    with st.container(border=True):
                        show_image(p)
                        st.markdown(f"**{p['name']}**")
                        st.markdown(f"<span class='price'>₹{p['price']}</span>", unsafe_allow_html=True)
                        left_stock = st.session_state.stock[p["id"]]
                        if left_stock == 0:
                            st.error("Out of stock")
                        else:
                            st.caption(f"{left_stock} in stock")
                        st.button("Add to cart", key=f"add_{p['id']}", on_click=add_to_cart,
                                  args=(p["id"],), disabled=(left_stock == 0), use_container_width=True)

    # ----- RIGHT: cart -----
    with right:
        st.subheader("🧺 Your Cart")
        if not st.session_state.cart:
            st.info("Your cart is empty. Add some products!")
        else:
            for pid, qty in list(st.session_state.cart.items()):
                p = get_product(pid)
                with st.container(border=True):
                    top_img, top_info = st.columns([1, 3])
                    with top_img:
                        show_image(p, width=56)
                    with top_info:
                        st.markdown(f"**{p['name']}**")
                        st.caption(f"₹{p['price']} × {qty} = ₹{p['price'] * qty}")
                    b_minus, b_qty, b_plus, b_del = st.columns([1, 1, 1, 2])
                    b_minus.button("−", key=f"dec_{pid}", on_click=decrease_qty, args=(pid,), use_container_width=True)
                    b_qty.markdown(f"<div class='qty'>{qty}</div>", unsafe_allow_html=True)
                    b_plus.button("+", key=f"inc_{pid}", on_click=add_to_cart, args=(pid,), use_container_width=True)
                    b_del.button("Remove", key=f"del_{pid}", on_click=remove_item, args=(pid,), use_container_width=True)


            st.markdown("---")
            # coupon
            cc1, cc2 = st.columns([3, 1])
            code_input = cc1.text_input("Coupon code", value=st.session_state.coupon,
                                        placeholder="e.g. SAVE10", label_visibility="collapsed")
            if cc2.button("Apply"):
                st.session_state.coupon = code_input.strip().upper()
                st.rerun()
            if st.session_state.coupon:
                if st.session_state.coupon in COUPONS:
                    st.success(f"Coupon {st.session_state.coupon} applied ✅")
                else:
                    st.error("Invalid coupon code ❌")

            # bill
            bill = calculate_bill()
            st.markdown("### 🧾 Bill Summary")
            st.write(f"Subtotal: ₹{bill['subtotal']:.2f}")
            if bill["auto_discount"] > 0:
                st.write(f"🎉 Auto Discount ({AUTO_DISCOUNT_PERCENT}%): -₹{bill['auto_discount']:.2f}")
            if bill["coupon_discount"] > 0:
                st.write(f"🏷️ Coupon Discount: -₹{bill['coupon_discount']:.2f}")
            st.write(f"GST ({int(GST_RATE * 100)}%): ₹{bill['gst']:.2f}")
            st.metric("Total to Pay", f"₹{bill['total']:.2f}")

            if bill["subtotal"] < AUTO_DISCOUNT_LIMIT:
                st.caption(f"Add ₹{AUTO_DISCOUNT_LIMIT - bill['subtotal']:.0f} more to get {AUTO_DISCOUNT_PERCENT}% off!")

            b1, b2 = st.columns(2)
            if b1.button("✅ Place Order", use_container_width=True, type="primary"):
                items = []
                for pid, qty in st.session_state.cart.items():
                    st.session_state.stock[pid] -= qty       # stock reduces after order
                    p = get_product(pid)
                    items.append({"Product": p["name"], "Qty": qty,
                                  "Price (₹)": int(p["price"]), "Amount (₹)": int(p["price"]) * qty})
                st.session_state.orders += 1
                st.session_state.order_history.append({
                    "no": st.session_state.orders,
                    "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                    "items": items,
                    "subtotal": bill["subtotal"],
                    "discount": bill["auto_discount"] + bill["coupon_discount"],
                    "gst": bill["gst"],
                    "total": bill["total"],
                    "receipt": make_receipt(bill),
                })
                st.session_state.just_ordered = True
                clear_cart()
                st.rerun()
            b2.button("Clear Cart", on_click=clear_cart, use_container_width=True)
            st.download_button("⬇️ Download Receipt", make_receipt(bill), file_name="receipt.txt",
                               use_container_width=True)

            # ML recommendations
            recs = recommend(list(st.session_state.cart.keys()))
            if recs:
                st.markdown("### 🤖 Customers also bought")
                for p in recs:
                    r0, r1, r2 = st.columns([1.2, 3, 1.6])
                    with r0:
                        show_image(p, width=48)
                    r1.write(f"{p['name']} - ₹{p['price']}")
                    r2.button("Add", key=f"rec_{p['id']}", on_click=add_to_cart, args=(p["id"],), use_container_width=True)

        if st.session_state.just_ordered:
            st.success(f"🎉 Order #{st.session_state.orders} placed successfully! See it in the 'My Orders' tab.")
            st.balloons()
            st.session_state.just_ordered = False


with tab_orders:
    st.subheader("📦 My Orders")
    history = st.session_state.order_history
    if not history:
        st.info("You haven't placed any orders yet. Add items to your cart and click 'Place Order'.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Orders placed", len(history))
        m2.metric("Total spent", f"₹{sum(o['total'] for o in history):.2f}")
        for o in reversed(history):                     # newest order first
            with st.expander(f"Order #{o['no']}   |   {o['time']}   |   ₹{o['total']:.2f}"):
                st.dataframe(pd.DataFrame(o["items"]), hide_index=True)
                st.write(f"Subtotal: ₹{o['subtotal']:.2f}")
                if o["discount"] > 0:
                    st.write(f"Discount: -₹{o['discount']:.2f}")
                st.write(f"GST: ₹{o['gst']:.2f}")
                st.markdown(f"**Total paid: ₹{o['total']:.2f}**")
                st.download_button("⬇️ Download Receipt", o["receipt"],
                                   file_name=f"receipt_order_{o['no']}.txt", key=f"receipt_{o['no']}")
