import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import questionary
import os
import time
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text
from fuzzywuzzy import process  
import re

console = Console()

use_cols_books = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']
use_cols_ratings = ['User-ID', 'ISBN', 'Book-Rating']
use_cols_users = ['User-ID', 'Age']

books = pd.read_csv('dataset/books.csv', sep=';', encoding='latin-1', usecols=use_cols_books, on_bad_lines='skip')
users = pd.read_csv('dataset/users.csv', sep=';', encoding='latin-1', usecols=use_cols_users, on_bad_lines='skip')
ratings = pd.read_csv('dataset/ratings.csv', sep=';', encoding='latin-1', usecols=use_cols_ratings, on_bad_lines='skip')

books.columns = ['ISBN', 'Title', 'Author', 'Year', 'Publisher']
ratings.columns = ['User_ID', 'ISBN', 'Book_Rating']
users.columns = ['User_ID', 'Age']

# Keep users who rated at least 50 books
user_counts = ratings['User_ID'].value_counts()
valid_users = user_counts[user_counts >= 50].index
ratings = ratings[ratings['User_ID'].isin(valid_users)]

# Keep books with at least 10 ratings
book_counts = ratings['ISBN'].value_counts()
valid_books = book_counts[book_counts >= 10].index
ratings = ratings[ratings['ISBN'].isin(valid_books)]

# Merge only relevant books and ratings
data = pd.merge(ratings, books, on='ISBN')

# Pivot table
book_pivot = data.pivot_table(index='Title', columns='User_ID', values='Book_Rating').fillna(0)

# Compute similarity matrix
book_similarity = cosine_similarity(book_pivot)

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

# Look up closest book
def find_closest_book(title):
    if not title or not isinstance(title, str):
        return None
        
    all_titles = book_pivot.index.tolist()
    lower_title = title.lower().strip()
    
    # Exact Match
    exact_matches = [t for t in all_titles if t.lower() == lower_title]
    if exact_matches:
        return exact_matches[0]
    
    # Substring Match (ignoring case)
    substring_matches = [t for t in all_titles if lower_title in t.lower()]
    if substring_matches:
        return min(substring_matches, key=len)  # Prefer shorter match
    
    # Word Boundary Match (ignoring punctuation)
    pattern = rf"\b{re.escape(lower_title)}\b"
    word_matches = [t for t in all_titles if re.search(pattern, t, re.IGNORECASE)]
    if word_matches:
        return word_matches[0]
    
    # Fuzzy Match (Fallback)
    result = process.extractOne(title, all_titles, scorer=process.fuzz.WRatio)
    return result[0] if result and result[1] > 70 else None


def recommend_books(book_title, n_recommendations=5):
    closest_title = find_closest_book(book_title)
    if not closest_title:
        return None

    # Get index of the book
    index = book_pivot.index.get_loc(closest_title)

    # Get similarity scores
    similarity_scores = list(enumerate(book_similarity[index]))

    # Sort books based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get indices of top similar books
    similar_books_indices = [i[0] for i in similarity_scores[1:n_recommendations+1]]

    # Get book titles safely
    similar_books = book_pivot.index.to_numpy()[similar_books_indices].tolist()

    return similar_books, closest_title

def show_recommendations(user_input):
    clear_screen()
    result = recommend_books(user_input)
    
    if not result:
        console.print(Panel.fit(
            f"[red]No close match found for '{user_input}'! Try another title.[/red]\n"
            f"[yellow]Example books: {', '.join(book_pivot.index[:3])}[/yellow]",
            border_style="red"
        ))
        return

    recommendations, matched_title = result

    if not matched_title:
        console.print(Panel.fit(
            f"[red]No exact match found for '{user_input}'! Try another title.[/red]",
            border_style="red"
        ))
        return
    
    if matched_title.lower() != user_input.lower():
        console.print(Panel.fit(
            f"[yellow]🔍 Showing results for: '[bold]{matched_title}[/bold]'[/yellow]",
            border_style="yellow"
        ))

    # Create recommendation table
    table = Table(
        title=f"📚 Recommended Books Similar to '[bold yellow]{matched_title}[/bold yellow]'", 
        box=box.ROUNDED,
        title_style="bold cyan",
        header_style="bold magenta",
        border_style="blue"
    )
    
    # Add columns
    table.add_column("#", style="cyan", justify="center")
    table.add_column("Title", style="yellow", min_width=30)
    table.add_column("Author", style="green")
    table.add_column("Year", style="magenta", justify="center")
    table.add_column("Avg Rating", style="blue", justify="center")

    # Get additional book details
    for idx, title in enumerate(recommendations, 1):
        book_data = data[data['Title'] == title]
        author = book_data['Author'].iloc[0]
        year = str(book_data['Year'].iloc[0])
        avg_rating = f"{book_data['Book_Rating'].mean():.1f} ★"
        
        table.add_row(
            str(idx),
            title,
            author,
            year,
            avg_rating
        )

    console.print(table)

def main():
    clear_screen()
    console.print(Panel.fit(
        Text("📖 AI Book Recommendation System 📖\n", style="bold magenta") +
        Text("🔍 Enter any book title (even partially) and I'll find similar ones!\n", style="cyan") +
        Text("💡 Type 'exit' to quit or 'examples' to see sample books", style="yellow"),
        border_style="bold green"
    ))

    while True:
        user_input = questionary.text("🔹 Enter a book title:").ask().strip()

        if user_input.lower() == "exit":
            clear_screen()
            console.print(Panel.fit("[bold red] Goodbye! Happy Reading! [/bold red]", border_style="red"))
            time.sleep(1)  
            break
            
        if user_input.lower() == "examples":
            clear_screen()
            sample_books = book_pivot.index[:5].tolist()
            console.print(Panel.fit(
                Text("📚 Sample Books in Database:\n", style="bold magenta") +
                Text("\n".join(f"• {book}" for book in sample_books), style="cyan"),
                border_style="blue"
            ))
            continue

        show_recommendations(user_input)

if __name__ == "__main__":
    main()