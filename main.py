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

console = Console()

# Load datasets
books = pd.read_csv('dataset/books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
users = pd.read_csv('dataset/users.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
ratings = pd.read_csv('dataset/ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

# Rename columns for easier access
books.columns = ['ISBN', 'Title', 'Author', 'Year', 'Publisher', 'Image_URL_S', 'Image_URL_M', 'Image_URL_L']
users.columns = ['User_ID', 'Location', 'Age']
ratings.columns = ['User_ID', 'ISBN', 'Book_Rating']

# Merge datasets
data = pd.merge(ratings, books, on='ISBN')

# Filter users who have rated at least 50 books
user_counts = data['User_ID'].value_counts()
data = data[data['User_ID'].isin(user_counts[user_counts >= 50].index)]

# Filter books that have at least 50 ratings
book_counts = data['Title'].value_counts()
data = data[data['Title'].isin(book_counts[book_counts >= 50].index)]

# Pivot table
book_pivot = data.pivot_table(index='Title', columns='User_ID', values='Book_Rating').fillna(0)

# Compute similarity matrix
book_similarity = cosine_similarity(book_pivot)


def clear_screen():
    """Clear the console screen (works for both Windows and Linux/Mac)."""
    os.system('cls' if os.name == 'nt' else 'clear')


def recommend_books(book_title, n_recommendations=5):
    if book_title not in book_pivot.index:
        return None  # Handle errors gracefully

    # Get index of the book
    index = book_pivot.index.get_loc(book_title)

    # Get similarity scores
    similarity_scores = list(enumerate(book_similarity[index]))

    # Sort books based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get indices of top similar books
    similar_books_indices = [i[0] for i in similarity_scores[1:n_recommendations+1]]

    # Get book titles safely
    similar_books = book_pivot.index.to_numpy()[similar_books_indices].tolist()

    return similar_books


def show_recommendations(book_name):
    clear_screen()
    recommendations = recommend_books(book_name)

    if recommendations is None:
        console.print(Panel.fit(f"[red]⚠️ Book '{book_name}' not found in our database! Try another title.[/red]", border_style="red"))
        return

    table = Table(title=f"Books Similar to '{book_name}'", box=box.DOUBLE_EDGE, title_style="bold cyan")
    table.add_column("No.", style="bold cyan", justify="center")
    table.add_column("Book Title", style="yellow")

    for idx, book in enumerate(recommendations, 1):
        table.add_row(str(idx), book)

    console.print(table)


def main():
    clear_screen()
    console.print(Panel.fit(
        Text("📖 AI Book Recommendation System 📖\n", style="bold magenta") +
        Text("🔍 Just Enter a book and I'll list similar ones!\n", style="cyan") +
        Text("💡 Type 'exit' anytime to quit.", style="yellow"),
        border_style="bold green"
    ))

    while True:
        book_name = questionary.text("🔹 Enter a book title:").ask()

        if book_name.lower() == "exit":
            clear_screen()
            console.print(Panel.fit("[bold red] Goodbye! Happy Reading! [/bold red]", border_style="red"))
            time.sleep(1)  
            break

        show_recommendations(book_name)


if __name__ == "__main__":
    main()
