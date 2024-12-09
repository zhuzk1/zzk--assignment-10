// Function to display results
function displayResults(results) {
    const resultsDiv = document.getElementById("results");
    const resultsContainer = document.getElementById("results-container");

    // 清除之前的结果
    resultsContainer.innerHTML = "";
    resultsDiv.style.display = "none";

    if (!results || results.length === 0) {
        resultsDiv.innerHTML = "<p>No results found.</p>";
        resultsDiv.style.display = "block";
        return;
    }

    results.forEach(result => {
        const resultDiv = document.createElement("div");
        resultDiv.style.textAlign = "center";
        resultDiv.style.margin = "10px";

        const img = document.createElement("img");
        img.src = `${window.location.origin}/results/${result.file_name}`;
        img.alt = "Search result";
        img.style.width = "150px";
        img.style.height = "auto"; 

        const similarity = document.createElement("p");
        similarity.textContent = `Similarity: ${result.similarity.toFixed(2)}`;

        resultDiv.appendChild(img);
        resultDiv.appendChild(similarity);
        resultsContainer.appendChild(resultDiv);
    });

    resultsDiv.style.display = "block";
}


// Handle Text Search
document.getElementById("text-search-form").addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent form submission

    const textQuery = document.getElementById("textQuery").value.trim();

    if (!textQuery) {
        alert("Please enter a text query.");
        return;
    }

    try {
        // Send the text query to the backend
        const response = await fetch("/search_text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: textQuery }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const results = await response.json();
        displayResults(results);
    } catch (error) {
        console.error("Error during text search:", error);
        alert("An error occurred while performing the text search.");
    }
});

// Handle Image Search
document.getElementById("image-search-form").addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent form submission

    const imageQuery = document.getElementById("imageQuery").files[0];
    const pcaComponents = document.getElementById("pcaComponents").value;

    if (!imageQuery) {
        alert("Please upload an image.");
        return;
    }

    try {
        const formData = new FormData();
        formData.append("image", imageQuery);
        if (pcaComponents) {
            formData.append("pca", pcaComponents);
        }

        // Send the image query to the backend
        const response = await fetch("/search_image", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const results = await response.json();
        displayResults(results);
    } catch (error) {
        console.error("Error during image search:", error);
        alert("An error occurred while performing the image search.");
    }
});

// Handle Combined Search
document.getElementById("combined-search-form").addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent form submission

    const combinedTextQuery = document.getElementById("combinedTextQuery").value.trim();
    const combinedImageQuery = document.getElementById("combinedImageQuery").files[0];
    const weight = document.getElementById("weight").value;

    if (!combinedTextQuery || !combinedImageQuery) {
        alert("Please provide both a text query and an image.");
        return;
    }

    try {
        const formData = new FormData();
        formData.append("query", combinedTextQuery);
        formData.append("image", combinedImageQuery);
        formData.append("weight", weight);

        // Send the combined query to the backend
        const response = await fetch("/search_hybrid", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const results = await response.json();
        displayResults(results);
    } catch (error) {
        console.error("Error during combined search:", error);
        alert("An error occurred while performing the combined search.");
    }
});
