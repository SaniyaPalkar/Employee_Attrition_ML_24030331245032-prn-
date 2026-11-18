
/* ==========================================
   1. HR RECOMMENDATIONS FUNCTION
========================================== */

function generateHRRecommendations(employee) {

    const recommendations = [];

    if (Number(employee.JobSatisfaction) <= 2) {
        recommendations.push(
            "Low job satisfaction: HR should discuss employee concerns and explore ways to improve job satisfaction."
        );
    }

    if (Number(employee.EnvironmentSatisfaction) <= 2) {
        recommendations.push(
            "Work environment: Review workplace conditions and identify possible improvements."
        );
    }

    if (Number(employee.WorkLifeBalance) <= 2) {
        recommendations.push(
            "Work-life balance: Consider flexible working arrangements and workload adjustments."
        );
    }

    if (employee.OverTime === "Yes") {
        recommendations.push(
            "Overtime: Review working hours and check whether the employee has an excessive workload."
        );
    }

    if (Number(employee.JobInvolvement) <= 2) {
        recommendations.push(
            "Job involvement: Discuss employee interests and opportunities to participate in meaningful work."
        );
    }

    if (Number(employee.YearsSinceLastPromotion) >= 5) {
        recommendations.push(
            "Career growth: Discuss career goals, training opportunities and possible advancement."
        );
    }

    if (Number(employee.DistanceFromHome) >= 20) {
        recommendations.push(
            "Commuting distance: Consider whether flexible work options or commuting support could help."
        );
    }

    let recommendationHTML = `
        <div class="hr-recommendations">
            <h3>HR Recommendations</h3>
    `;

    if (recommendations.length > 0) {

        recommendationHTML += "<ul>";

        recommendations.forEach(item => {
            recommendationHTML += `<li>${item}</li>`;
        });

        recommendationHTML += "</ul>";

    } else {

        recommendationHTML += `
            <p>
                No obvious review areas were identified by these simple rules.
                HR should still discuss employee needs and concerns.
            </p>
        `;

    }

    recommendationHTML += `
            <p class="recommendation-note">
                These are rule-based suggestions for HR review.
                They are not confirmed reasons for employee attrition.
            </p>
        </div>
    `;

    return recommendationHTML;
}


/* ==========================================
   2. EMPLOYEE ATTRITION PREDICTION
========================================== */

const predictionForm = document.getElementById("predictionForm");

predictionForm.addEventListener("submit", async function(event) {

    event.preventDefault();

    const resultBox = document.getElementById("predictionResult");
    const submitButton = predictionForm.querySelector("button");

    submitButton.disabled = true;
    submitButton.textContent = "Predicting...";

    resultBox.hidden = false;
    resultBox.className = "result-box";

    resultBox.innerHTML = `
        <h3>Prediction Result</h3>
        <div id="predictionDetails">
            Analyzing employee information...
        </div>
    `;

    const formData = new FormData(predictionForm);
    const employee = {};

    for (const [key, value] of formData.entries()) {

        if (key !== "model") {
            employee[key] = value.trim();
        }

    }

    // Convert numeric employee fields into numbers

    const numericFields = [
        "Age", "DailyRate", "DistanceFromHome", "Education",
        "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
        "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
        "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StockOptionLevel",
        "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
        "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager"
    ];

    numericFields.forEach(field => {
        employee[field] = Number(employee[field]);
    });

    const selectedModel = formData.get("model");

    try {

        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                employee: employee,
                model: selectedModel
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Prediction failed.");
        }

        const predictionDetails = document.getElementById(
            "predictionDetails"
        );

        const probability = Number(result.attrition_probability);
        const isLeaving = result.prediction === "Likely to Leave";

        resultBox.className = `result-box ${
            isLeaving ? "result-danger" : "result-safe"
        }`;

        predictionDetails.innerHTML = `
            <div class="prediction-status">
                <span>${isLeaving ? "⚠️" : "✅"}</span>
                <h2>${result.prediction}</h2>
            </div>

            <div class="prediction-probability">
                <p>Attrition Probability</p>
                <strong>${probability.toFixed(2)}%</strong>
            </div>

            <div class="probability-track">
                <div class="probability-fill"
                     style="width: ${Math.min(100, Math.max(0, probability))}%">
                </div>
            </div>

            <div class="prediction-info">
                <div>
                    <span>Selected Model</span>
                    <strong>${result.model}</strong>
                </div>

                <div>
                    <span>K-Means Group</span>
                    <strong>Cluster ${Number(result.cluster) + 1}</strong>
                </div>
            </div>

            <p class="prediction-disclaimer">
                This is a model-based estimate, not a certainty or a
                decision about an employee.
            </p>
        `;

        // Display HR recommendations below the prediction

        predictionDetails.insertAdjacentHTML(
            "afterend",
            generateHRRecommendations(employee)
        );

    } catch (error) {

        resultBox.className = "result-box";
        resultBox.textContent = "Error: " + error.message;

    } finally {

        submitButton.disabled = false;
        submitButton.textContent = "Predict Employee Attrition";

    }

});


/* ==========================================
   3. ANN VS SVM MODEL COMPARISON
========================================== */

async function loadComparison() {

    const message = document.getElementById("comparisonMessage");
    const resultsContainer = document.getElementById("comparisonResults");

    try {

        const response = await fetch("/api/comparison");

        if (!response.ok) {
            throw new Error("Could not load comparison results.");
        }

        const results = await response.json();

        const ann = results.find(item => item.Model === "ANN");
        const svm = results.find(item => item.Model === "SVM");

        if (!ann || !svm) {
            throw new Error("ANN or SVM results are missing.");
        }

        message.textContent = "Model evaluation results";

        let html = `
            <div class="comparison-table">
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>ANN</th>
                            <th>SVM</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        const metrics = [
            ["Accuracy", "Accuracy"],
            ["Precision", "Precision"],
            ["Recall", "Recall"],
            ["F1 Score", "F1 Score"],
            ["ROC AUC", "ROC AUC"]
        ];

        metrics.forEach(([label, key]) => {

            html += `
                <tr>
                    <td>${label}</td>
                    <td>${(ann[key] * 100).toFixed(2)}%</td>
                    <td>${(svm[key] * 100).toFixed(2)}%</td>
                </tr>
            `;

        });

        html += `
                    </tbody>
                </table>
            </div>

            <h3 class="form-heading">Accuracy Comparison</h3>
        `;

        results.forEach(item => {

            const accuracy = (item.Accuracy * 100).toFixed(2);

            html += `
                <div class="accuracy-item">
                    <div class="accuracy-label">
                        <span>${item.Model}</span>
                        <strong>${accuracy}%</strong>
                    </div>

                    <div class="accuracy-track">
                        <div class="accuracy-fill"
                             style="width: ${accuracy}%">
                        </div>
                    </div>
                </div>
            `;

        });

        resultsContainer.innerHTML = html;

    } catch (error) {

        message.textContent = error.message;

    }

}

loadComparison();


/* ==========================================
   4. K-MEANS CLUSTER ANALYSIS
========================================== */

async function loadClusters() {

    const message = document.getElementById("clusterMessage");
    const container = document.getElementById("clusterResults");

    try {

        const response = await fetch("/api/clusters");

        if (!response.ok) {
            throw new Error("Could not load cluster data.");
        }

        const data = await response.json();

        if (data.status !== "success") {
            throw new Error(data.error || "Cluster data unavailable.");
        }

        message.textContent = "Employee group analysis";

        const profiles = data.employee_profiles;
        const attrition = data.attrition_profiles;
        const overtime = data.overtime_profiles;

        let html = '<div class="cluster-grid">';

        profiles.forEach(profile => {

            const id = profile.Cluster;

            const attritionData = attrition.find(
                item => item.Cluster === id
            );

            const overtimeData = overtime.find(
                item => item.Cluster === id
            );

            const descriptions = {
                0: "Mid-career employees with moderate income and experience.",
                1: "Relatively younger employees with lower average income and experience.",
                2: "More experienced employees with higher average income."
            };

            html += `
                <div class="cluster-card">

                    <h3>Cluster ${id + 1}</h3>

                    <p class="cluster-description">
                        ${descriptions[id] || "Employee group identified by K-Means."}
                    </p>

                    <div class="cluster-stat">
                        <span>Average Age</span>
                        <strong>${profile.Age.toFixed(1)} years</strong>
                    </div>

                    <div class="cluster-stat">
                        <span>Monthly Income</span>
                        <strong>${Math.round(profile.MonthlyIncome).toLocaleString("en-IN")}</strong>
                    </div>

                    <div class="cluster-stat">
                        <span>Job Satisfaction</span>
                        <strong>${profile.JobSatisfaction.toFixed(2)} / 4</strong>
                    </div>

                    <div class="cluster-stat">
                        <span>Years at Company</span>
                        <strong>${profile.YearsAtCompany.toFixed(1)} years</strong>
                    </div>

                    <div class="cluster-stat">
                        <span>Total Working Experience</span>
                        <strong>${profile.TotalWorkingYears.toFixed(1)} years</strong>
                    </div>

                    <div class="cluster-divider"></div>

                    <h4>Attrition Analysis</h4>

                    <div class="cluster-stat">
                        <span>Employees Who Left</span>
                        <strong>${attritionData["Left (%)"].toFixed(2)}%</strong>
                    </div>

                    <div class="cluster-stat">
                        <span>Employees Who Stayed</span>
                        <strong>${attritionData["Stayed (%)"].toFixed(2)}%</strong>
                    </div>

                    <div class="cluster-divider"></div>

                    <h4>Overtime Distribution</h4>

                    <div class="cluster-stat">
                        <span>Overtime: Yes</span>
                        <strong>${overtimeData.Yes.toFixed(2)}%</strong>
                    </div>

                    <div class="cluster-stat">
                        <span>Overtime: No</span>
                        <strong>${overtimeData.No.toFixed(2)}%</strong>
                    </div>

                </div>
            `;

        });

        html += "</div>";

        container.innerHTML = html;


        // CHART 1: Attrition Rate by Cluster

        document.getElementById("attritionChart").innerHTML = `
            <div class="chart-bars">

                ${attrition.map(item => `
                    <div class="chart-bar-item">

                        <strong>${item["Left (%)"].toFixed(2)}%</strong>

                        <div class="chart-bar-track">
                            <div class="chart-bar-fill"
                                 style="height: ${item["Left (%)"]}%;"></div>
                        </div>

                        <span>Cluster ${item.Cluster + 1}</span>

                    </div>
                `).join("")}

            </div>
        `;


        // CHART 2: Average Monthly Income

        document.getElementById("incomeChart").innerHTML = `
            <div class="chart-bars">

                ${profiles.map(item => `
                    <div class="chart-bar-item">

                        <strong>${Math.round(item.MonthlyIncome).toLocaleString("en-IN")}</strong>

                        <div class="chart-bar-track">
                            <div class="chart-bar-fill income-fill"
                                 style="height: ${(item.MonthlyIncome / 12000) * 100}%;"></div>
                        </div>

                        <span>Cluster ${item.Cluster + 1}</span>

                    </div>
                `).join("")}

            </div>
        `;


        // CHART 3: Overtime Distribution

        document.getElementById("overtimeChart").innerHTML = `
            ${overtime.map(item => `

                <div class="overtime-row">

                    <strong>Cluster ${item.Cluster + 1}</strong>

                    <div class="overtime-track">

                        <div class="overtime-yes"
                             style="width: ${item.Yes}%;"></div>

                        <div class="overtime-no"
                             style="width: ${item.No}%;"></div>

                    </div>

                    <div class="overtime-legend">
                        <span>Yes: ${item.Yes.toFixed(2)}%</span>
                        <span>No: ${item.No.toFixed(2)}%</span>
                    </div>

                </div>

            `).join("")}
        `;

    } catch (error) {

        message.textContent = error.message;

    }

}

loadClusters();


/* ==========================================
   5. SIDEBAR NAVIGATION
========================================== */

const sidebarLinks = document.querySelectorAll(".sidebar > a");

const pageSections = document.querySelectorAll(
    ".main-content section[id]"
);


// Smooth scrolling when clicking sidebar links

sidebarLinks.forEach(link => {

    link.addEventListener("click", function(event) {

        const targetId = this.getAttribute("href");
        const targetSection = document.querySelector(targetId);

        if (targetSection) {

            event.preventDefault();

            targetSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

            sidebarLinks.forEach(item => {
                item.classList.remove("active");
            });

            this.classList.add("active");

            history.replaceState(null, "", targetId);

        }

    });

});


// Update active sidebar link while scrolling

const sectionObserver = new IntersectionObserver(

    entries => {

        entries.forEach(entry => {

            if (entry.isIntersecting) {

                const currentId = entry.target.id;

                sidebarLinks.forEach(link => {

                    link.classList.toggle(
                        "active",
                        link.getAttribute("href") === `#${currentId}`
                    );

                });

            }

        });

    },

    {
        threshold: 0,
        rootMargin: "-20% 0px -65% 0px"
    }

);

pageSections.forEach(section => {
    sectionObserver.observe(section);
});
