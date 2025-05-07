import streamlit as st
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Callable
import time
import os
from dotenv import load_dotenv
import base64

from llm_service import LLMService
from vendor_manager import VendorManager, VendorDetail
from utils import get_css, get_skeleton_card_html

# Load environment variables
load_dotenv()

# Function to load and display the logo
def get_logo_base64():
    with open("logo.png", "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

# Custom CSS for hiding sidebar in certain steps
def get_sidebar_visibility_css(show_sidebar):
    if not show_sidebar:
        return """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """
    return ""

# Page configuration
st.set_page_config(
    page_title="Supervity | Vendor Research Agent",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme with custom CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] > div:first-child {
        background-color: #ffffff;
    }
    .stApp {
        background-color: #ffffff;
    }
    html, body, [class*="css"] {
        color: rgb(49, 51, 63);
        background-color: #ffffff;
    }
    .stMarkdown, .stTextInput > div > div > input {
        color: rgb(49, 51, 63);
    }
    button[kind="primary"] {
        background-color: rgb(255, 75, 75);
        color: rgb(255, 255, 255);
    }
    button[data-testid="baseButton-secondary"] {
        background-color: white;
        color: rgb(49, 51, 63);
    }
    span[data-baseweb="tag"] {
        background-color: rgba(255, 75, 75, 0.2);
    }
    div[role="radiogroup"] label {
        color: rgb(49, 51, 63);
    }
    div[data-testid="stExpander"] {
        border: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# Inject custom CSS
st.markdown(get_css(), unsafe_allow_html=True)

# Determine if sidebar should be shown
show_sidebar = st.session_state.get("step", 1) == 3
st.markdown(get_sidebar_visibility_css(show_sidebar), unsafe_allow_html=True)

# Display logo in sidebar only in step 3
if show_sidebar:
    logo_base64 = get_logo_base64()
    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" style="width: 80%; max-width: 200px; height: auto; object-fit: contain;">
        </div>
        """,
        unsafe_allow_html=True
    )

# Get logo for main display
logo_base64 = get_logo_base64()

# Title and description - improved professional layout
st.markdown(
    f"""
    <style>
        .header-container {{
            display: flex;
            flex-direction: column;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid #e5e7eb;
        }}
        .title-row {{
            display: flex;
            align-items: center;
        }}
        .logo-brand {{
            display: flex;
            align-items: center;
        }}
        .brand-name {{
            font-size: 2rem;
            font-weight: 600;
            color: #111827;
            margin-left: 12px;
        }}
        .agent-title {{
            margin: 0 0 0 30px;
            padding: 0;
            font-size: 2.5rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.2;
        }}
        .description {{
            margin: 12px 0 0 0;
            padding: 0;
            font-size: 1.2rem;
            color: #4b5563;
            max-width: 800px;
        }}
        @media (max-width: 768px) {{
            .title-row {{
                flex-direction: column;
                align-items: flex-start;
            }}
            .agent-title {{
                margin: 15px 0 0 0;
                font-size: 2rem;
            }}
        }}
    </style>
    <div class="header-container">
        <div class="title-row">
            <div class="logo-brand">
                <img src="data:image/png;base64,{logo_base64}" style="height: 60px; width: auto;">
                <span class="brand-name"></span>
            </div>
            <h1 class="agent-title">Vendor Research Agent</h1>
        </div>
        <p class="description">Find, research, and discover real vendors for any industry or category.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1  # 1: Disambiguation, 2: Config, 3: Results
    
if "term" not in st.session_state:
    st.session_state.term = ""
    
if "interpretations" not in st.session_state:
    st.session_state.interpretations = []
    
if "selected_interpretation" not in st.session_state:
    st.session_state.selected_interpretation = None
    
if "selected_interpretation_index" not in st.session_state:
    st.session_state.selected_interpretation_index = None
    
if "vendors" not in st.session_state:
    st.session_state.vendors = []
    
if "enriched_vendors" not in st.session_state:
    st.session_state.enriched_vendors = []
    
if "loading" not in st.session_state:
    st.session_state.loading = False
    
if "progress" not in st.session_state:
    st.session_state.progress = 0
    
if "mix" not in st.session_state:
    st.session_state.mix = {"manufacturer": 40, "distributor": 30, "retailer": 30}
    
if "country" not in st.session_state:
    st.session_state.country = "United States"
    
if "region" not in st.session_state:
    st.session_state.region = ""
    
if "vendor_count" not in st.session_state:
    st.session_state.vendor_count = 20
    
if "show_advanced" not in st.session_state:
    st.session_state.show_advanced = False

# Initialize LLM service and Vendor Manager
@st.cache_resource
def get_llm_service():
    return LLMService()

@st.cache_resource
def get_vendor_manager(_llm_service):
    return VendorManager(_llm_service)

llm_service = get_llm_service()
vendor_manager = get_vendor_manager(llm_service)

# Callback functions for navigation
def go_to_step(step):
    st.session_state.step = step
    
def select_interpretation(index):
    if 0 <= index < len(st.session_state.interpretations):
        st.session_state.selected_interpretation = st.session_state.interpretations[index]
        st.session_state.selected_interpretation_index = index
        st.session_state.step = 2  # Directly set step to 2
    
def start_vendor_search():
    st.session_state.loading = True
    st.session_state.vendors = []
    st.session_state.enriched_vendors = []
    st.session_state.step = 3  # Directly set step to avoid rerun issues

def update_progress(current, total):
    """Update the progress bar during vendor enrichment."""
    st.session_state.progress = current / total

def toggle_advanced():
    st.session_state.show_advanced = not st.session_state.show_advanced

def update_mix(manufacturer_pct, distributor_pct, retailer_pct):
    """Update mix percentages and normalize them to sum to 100%"""
    total = manufacturer_pct + distributor_pct + retailer_pct
    if total > 0:
        manufacturer_pct = round(manufacturer_pct / total * 100)
        distributor_pct = round(distributor_pct / total * 100)
        retailer_pct = 100 - manufacturer_pct - distributor_pct  # Ensure total is exactly 100%
    
    st.session_state.mix = {
        "manufacturer": manufacturer_pct,
        "distributor": distributor_pct,
        "retailer": retailer_pct
    }
    return manufacturer_pct, distributor_pct, retailer_pct

# Function to run async code
def run_async(func, *args, **kwargs):
    """Run an async function from Streamlit's synchronous environment."""
    result = []
    
    async def run_and_capture():
        r = await func(*args, **kwargs)
        result.append(r)
    
    # Always create a new event loop to avoid coroutine reuse issues
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_and_capture())
    finally:
        loop.close()
    
    if result:
        return result[0]
    
    return None

# Main app flow
if st.session_state.step == 1:
    # Step 1: Disambiguation
    st.header("Step 1: What are you looking for?")
    
    term = st.text_input("Enter a product, service, or industry", 
                        value=st.session_state.term,
                        placeholder="e.g., solar panels, IT consulting, 3D printing")
    
    if st.button("Find Vendors", key="find_vendors_btn"):
        if term:
            with st.spinner("Analyzing your search..."):
                st.session_state.term = term
                interpretations = run_async(llm_service.disambiguate_term, term)
                st.session_state.interpretations = interpretations
                
            st.subheader("Select the most relevant interpretation:")
            
            cols = st.columns(len(interpretations))
            for i, interp in enumerate(interpretations):
                with cols[i]:
                    st.markdown(f"### {interp['interpretation']}")
                    st.markdown(interp['description'])
                    if st.button("Select", key=f"select_interp_{i}", on_click=select_interpretation, args=(i,)):
                        pass  # The on_click function will handle the action
        else:
            st.error("Please enter a search term")

    # Check if interpretations exist but no selection made yet
    elif st.session_state.interpretations and st.session_state.selected_interpretation_index is None:
        st.subheader("Select the most relevant interpretation:")
        
        cols = st.columns(len(st.session_state.interpretations))
        for i, interp in enumerate(st.session_state.interpretations):
            with cols[i]:
                st.markdown(f"### {interp['interpretation']}")
                st.markdown(interp['description'])
                if st.button("Select", key=f"select_interp_session_{i}", on_click=select_interpretation, args=(i,)):
                    pass  # The on_click function will handle the action

elif st.session_state.step == 2:
    # Step 2: Configuration
    st.header("Step 2: Configure Your Vendor Search")
    
    # Display selected interpretation
    st.markdown(f"**Search term:** {st.session_state.selected_interpretation['interpretation']}")
    st.markdown(f"*{st.session_state.selected_interpretation['description']}*")
    
    # Vendor count
    count = st.slider("How many vendors do you need?", min_value=5, max_value=100, value=st.session_state.vendor_count, step=5)
    st.session_state.vendor_count = count  # Store the slider value in session state
    
    # Country and Region selection
    st.subheader("Location")
    
    # List of countries (simplified list)
    countries = [
        "United States", "Canada", "United Kingdom", "Australia", "Germany", 
        "France", "Japan", "China", "India", "Brazil", "Other"
    ]
    
    country = st.selectbox("Country", options=countries, index=countries.index(st.session_state.country) if st.session_state.country in countries else 0)
    
    # Region field (optional)
    region = st.text_input("Region/State (Optional)", value=st.session_state.region, placeholder="e.g., California, Ontario, etc.")
    
    # Save to session state
    st.session_state.country = country
    st.session_state.region = region
    
    # Advanced Options in an expander
    st.button("Advanced Options", key="toggle_advanced_btn", on_click=toggle_advanced)
    
    if st.session_state.show_advanced:
        with st.expander("Business Type Mix", expanded=True):
            st.markdown("Adjust the percentage of each business type:")
            
            # Get current mix values from session state
            current_manufacturer = st.session_state.mix.get("manufacturer", 40)
            current_distributor = st.session_state.mix.get("distributor", 30)
            current_retailer = st.session_state.mix.get("retailer", 30)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                manufacturer_pct = st.slider("Manufacturers (%)", 0, 100, current_manufacturer, 5, key="manufacturer_slider")
            
            with col2:
                distributor_pct = st.slider("Distributors (%)", 0, 100, current_distributor, 5, key="distributor_slider")
            
            with col3:
                retailer_pct = st.slider("Retailers (%)", 0, 100, current_retailer, 5, key="retailer_slider")
            
            # Normalize percentages to sum to 100% and update session state
            manufacturer_pct, distributor_pct, retailer_pct = update_mix(manufacturer_pct, distributor_pct, retailer_pct)
            
            # Display normalized percentages
            st.markdown(f"Adjusted mix: Manufacturers ({manufacturer_pct}%), "
                      f"Distributors ({distributor_pct}%), Retailers ({retailer_pct}%)")
    
    # Buttons for navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", key="back_to_disambiguation_btn"):
            go_to_step(1)
    
    with col2:
        if st.button("Find Vendors →", key="start_search_btn", on_click=start_vendor_search):
            pass  # The on_click function will handle the action

elif st.session_state.step == 3:
    # Step 3: Results
    st.header("Step 3: Found Vendors")
    
    # Add a back button at the top for better navigation
    if st.button("← Start New Search", key="main_new_search"):
        st.session_state.clear()
        st.rerun()
    
    # Setup sidebar filters - now only visible in step 3
    if show_sidebar:
        st.sidebar.header("Filters")
    
    # Create vendor search and enrichment if needed
    if st.session_state.loading:
        
        if not st.session_state.vendors:
            # Find vendors
            term = st.session_state.selected_interpretation["interpretation"]
            mix = st.session_state.mix
            count = st.session_state.vendor_count  # Use the stored vendor count
            country = st.session_state.country
            region = st.session_state.region
            
            with st.spinner(f"Searching for {count} vendors..."):
                # Display skeleton loaders while waiting
                st.markdown(f"Finding {count} vendors matching your criteria...")
                for _ in range(min(5, count)):
                    st.markdown(get_skeleton_card_html(), unsafe_allow_html=True)
                
                # Find vendor names
                vendor_names = run_async(vendor_manager.find_vendors, term, count, mix, country, region)
                st.session_state.vendors = [(name, next((k for k, v in mix.items() if v > 0), "general")) 
                                          for name in vendor_names]
                st.rerun()
        
        elif not st.session_state.enriched_vendors:
            # Research vendors
            term = st.session_state.selected_interpretation["interpretation"]
            country = st.session_state.country
            region = st.session_state.region
            vendor_count = len(st.session_state.vendors)
            
            # Progress bar
            progress_bar = st.progress(0)
            st.markdown(f"Researching information for {vendor_count} vendors...")
            
            # Display skeleton loaders while waiting
            for _ in range(min(5, len(st.session_state.vendors))):
                st.markdown(get_skeleton_card_html(), unsafe_allow_html=True)
            
            # Update progress callback
            def update_enrichment_progress(current, total):
                progress = current / total
                progress_bar.progress(progress)
            
            # Research vendors
            enriched_vendors = run_async(
                vendor_manager.research_vendors_batch,
                st.session_state.vendors,
                term,
                country,
                region,
                with_progress_callback=update_enrichment_progress
            )
            
            st.session_state.enriched_vendors = enriched_vendors
            st.session_state.loading = False
            st.rerun()
    
    # Show filter options if we have results
    if st.session_state.enriched_vendors:
        vendors = st.session_state.enriched_vendors
        
        # Sidebar filters
        if show_sidebar:
            min_score, max_score = st.sidebar.slider(
                "Relevance Score", 
                min_value=1, 
                max_value=10,
                value=(1, 10)
            )
            
            # Get simplified business types (standardize to main categories)
            def simplify_business_type(business_type):
                if 'manufacturer' in business_type.lower():
                    return 'Manufacturer'
                elif 'distributor' in business_type.lower() or 'supplier' in business_type.lower():
                    return 'Distributor/Supplier'
                elif 'retailer' in business_type.lower() or 'seller' in business_type.lower():
                    return 'Retailer'
                else:
                    return business_type
            
            # Map vendors to simplified business types
            for vendor in vendors:
                vendor.simplified_type = simplify_business_type(vendor.business_type)
            
            # Get unique simplified business types
            simplified_types = sorted(list(set(v.simplified_type for v in vendors)))
            
            selected_types = st.sidebar.multiselect(
                "Business Types",
                options=simplified_types,
                default=simplified_types
            )
            
            # Sorting options
            sort_options = {
                "Relevance (High to Low)": lambda v: -v.relevance_score,
                "Relevance (Low to High)": lambda v: v.relevance_score,
                "Name (A-Z)": lambda v: v.name,
                "Name (Z-A)": lambda v: v.name[::-1],
            }
            
            sort_by = st.sidebar.selectbox(
                "Sort by",
                options=list(sort_options.keys()),
                index=0
            )
            
            # Export options
            if st.sidebar.button("Export Results (CSV)", key="export_csv_btn"):
                # Create CSV content
                import csv
                import io
                
                # Create a string buffer
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    "Name", "Business Type", "Relevance Score", "Website", 
                    "Contact", "Description", "Specializations"
                ])
                
                # Filter vendors first
                filtered_vendors = [
                    v for v in vendors
                    if v.relevance_score >= min_score
                    and v.relevance_score <= max_score
                    and v.simplified_type in selected_types
                ]
                
                # Sort vendors
                filtered_vendors.sort(key=sort_options[sort_by])
                
                # Write vendor data
                for vendor in filtered_vendors:
                    writer.writerow([
                        vendor.name,
                        vendor.business_type,
                        vendor.relevance_score,
                        vendor.website,
                        vendor.contact,
                        vendor.description,
                        ", ".join(vendor.specializations)
                    ])
                
                # Create download link
                csv_data = output.getvalue()
                
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"vendors_{st.session_state.term}_{len(filtered_vendors)}.csv",
                    mime="text/csv"
                )
            
            if st.sidebar.button("← Start New Search", key="sidebar_new_search"):
                st.session_state.clear()
                st.rerun()
                
            # Filter vendors
            filtered_vendors = [
                v for v in vendors
                if v.relevance_score >= min_score
                and v.relevance_score <= max_score
                and v.simplified_type in selected_types
            ]
            
            # Sort vendors
            filtered_vendors.sort(key=sort_options[sort_by])
        else:
            # If sidebar is not shown, use all vendors without filtering
            filtered_vendors = vendors
            # Sort by relevance by default
            filtered_vendors.sort(key=lambda v: -v.relevance_score)
        
        # Display results count
        st.markdown(f"Found **{len(filtered_vendors)}** vendors matching your criteria.")
        
        # Display vendor cards
        for vendor in filtered_vendors:
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    # Logo placeholder (gray square with first letter)
                    st.markdown(f"""
                    <div style="width: 80px; height: 80px; background-color: #e9ecef; 
                                border-radius: 8px; display: flex; align-items: center; 
                                justify-content: center; font-size: 32px; font-weight: bold; 
                                color: #495057;">
                        {vendor.name[0]}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Determine score class
                    score_class = "score-high" if vendor.relevance_score >= 7 else \
                                "score-medium" if vendor.relevance_score >= 4 else "score-low"
                    
                    st.markdown(f"""
                    <div class="vendor-card">
                        <div>
                            <span class="vendor-name">{vendor.name}</span>
                            <span class="vendor-score {score_class}">{vendor.relevance_score}/10</span>
                            <div class="vendor-info">{vendor.business_type}</div>
                        </div>
                        <div class="vendor-description">{vendor.description}</div>
                        <div>
                            <a href="{vendor.website}" target="_blank" class="vendor-website">{vendor.website}</a>
                        </div>
                        <div class="vendor-contact">{vendor.contact}</div>
                        <div>
                            {"".join([f'<span class="vendor-specialization">{s}</span>' for s in vendor.specializations])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Show back button if no results yet
        if st.button("← Back to Configuration"):
            go_to_step(2)

# Run the application
if __name__ == "__main__":
    # This is already being run by Streamlit
    pass 