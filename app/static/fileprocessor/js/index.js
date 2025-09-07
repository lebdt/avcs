let fileId = null;
let socket = null;
let columnData = {
  all: [],
  numeric: [],
  categorical: [],
};
let selectedColumn = null;
let columnModalInstance = null;

let responseBestMatch = null;

const uploadForm = document.getElementById("upload-form");
const uploadStatus = document.getElementById("upload-status");
const uploadBtn = document.getElementById("upload-btn-submit");
const fileError = document.getElementById("file-error");
const loader = document.getElementById("loader");
const chatContainer = document.getElementById("chat-container");
const queryInput = document.getElementById("query-input");
const sendQueryBtn = document.getElementById("send-query");
const dataSection = document.getElementById("data-section");
const previewContent = document.getElementById("preview-content");
const statsContent = document.getElementById("stats-content");
const queryResultsContent = document.getElementById("query-results-content");
const overviewPaneContent = document.getElementById("overview-text");

function addMessage(message, isUser = false) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `chat-message ${isUser ? "user-message" : "system-message"}`;
  msgDiv.innerHTML = message;
  chatContainer.appendChild(msgDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function populateColumnData(stats) {
  console.log(stats.column_names);
  console.log(stats.categorical_columns);
  console.log(stats.numeric_columns);
  stats.column_names.forEach((column) => {
    columnData.all.push(column);
  });
  stats.numeric_columns.forEach((column) => {
    columnData.numeric.push(column);
  });
  stats.categorical_columns.forEach((column) => {
    columnData.categorical.push(column);
  });
}

function populateColumnSelectors(columns) {
  const predictorsSelect = document.getElementById("predictors-select");
  const responseSelect = document.getElementById("response-select");
  const pairResponseSelect = document.getElementById("pair-response-select");

  predictorsSelect.innerHTML = "";

  responseSelect.innerHTML = '<option value="">None</option>';

  columns.forEach((column) => {
    const predictorOption = document.createElement("option");
    predictorOption.value = column;
    predictorOption.textContent = column;
    predictorsSelect.appendChild(predictorOption);

    const responseOption = document.createElement("option");
    responseOption.value = column;
    responseOption.textContent = column;
    if (responseOption.value == responseBestMatch) {
      responseOption.selected = "selected";
    }
    responseSelect.appendChild(responseOption);
  });

  pairResponseSelect.innerHTML = '<option value="">None</option>';

  columns.forEach((column) => {
    const pairResponseOption = document.createElement("option");
    pairResponseOption.value = column;
    pairResponseOption.textContent = column;
    if (pairResponseOption.value == responseBestMatch) {
      pairResponseOption.selected = "selected";
    }
    pairResponseSelect.appendChild(pairResponseOption);
  });

}

function showLoader() {
  uploadStatus.classList.add("hidden");
  loader.classList.remove("hidden");
}

function hideLoader() {
  loader.classList.add("hidden");
  uploadStatus.classList.remove("hidden");
}

function enableChat() {
  queryInput.disabled = false;
  sendQueryBtn.disabled = false;
  addMessage(
    "Your CSV is loaded. Check the <b>Overview Pane</b> for quick insights.",
  );
}

function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/csv/`;

  socket = new WebSocket(wsUrl);

  socket.onopen = function (e) {
    console.log("WebSocket connection established");
  };

  socket.onmessage = function (e) {
    hideLoader();
    const data = JSON.parse(e.data);

    if (data.type === "csv_processed") {
      handleCSVProcessed(data);
    } else if (data.type === "query_result") {
      handleQueryResult(data);
    } else if (data.type === "analysis_result") {
      handleAnalysisResult(data);
    } else if (data.type === "column_added") {
      handleColumnAdded(data);
    } else if (data.type === "column_modified") {
      handleColumnModified(data);
    } else if (data.type === "column_deleted") {
      handleColumnDeleted(data);
    } else if (data.type === "pairs_tested") {
      handlePairTest(data);
    } else if (data.type === "error") {
      addMessage(`<span style="color:red;">Error: ${data.message}</span>`);
    }
  };

  socket.onclose = function (e) {
    console.log("WebSocket connection closed");
  };

  socket.onerror = function (e) {
    hideLoader();
    addMessage(
      `<span style="color:red;">Error: Could not connect to the server.</span>`,
    );
    console.error("WebSocket error:", e);
  };
}

function runStats(data) {
  columnData.all = data.stats.column_names;
  columnData.numeric = data.stats.numeric_columns;
  columnData.categorical = data.stats.categorical_columns;

  let statsHtml = `<div class="card-body">
                    <h5>Dataset</h5>
                    <p>Rows: ${data.stats.rows}</p>
                    <p>Columns: ${data.stats.columns}</p>
                    <h5>Columns</h5>
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>Column Name</th>
                                <th>Data Type</th>
                                ${Object.keys(data.stats.means).length > 0 ? "<th>Mean</th>" : ""}
                                ${Object.keys(data.stats.stddev).length > 0 ? "<th>Std. Dev.</th>" : ""}
                            </tr>
                        </thead>
                        <tbody>`;

  data.stats.column_names.forEach((col) => {
    statsHtml += `<tr>
                        <td>${col}</td>
                        <td>${data.stats.dtypes[col]}</td>
                        ${
                          Object.keys(data.stats.means).length > 0
                            ? `<td>${data.stats.means[col] !== undefined ? data.stats.means[col].toFixed(2) : "-"}</td>`
                            : ""
                        }
                        ${
                          Object.keys(data.stats.stddev).length > 0
                            ? `<td>${data.stats.stddev[col] !== undefined ? data.stats.stddev[col].toFixed(2) : "-"}</td>`
                            : ""
                        }
                    </tr>`;
  });

  statsHtml += `</tbody></table></div>`;
  statsContent.innerHTML = statsHtml;
}

function handleCSVProcessed(data) {
  if (data.success) {
    previewContent.innerHTML = data.preview;

    runStats(data);

    dataSection.classList.remove("hidden");

    overviewPaneContent.innerHTML = data.overview;

    responseBestMatch = data.default_response;

    updateColumnList(data);

    enableChat();

    document.getElementById("analysis-tab").classList.remove("disabled");
  } else {
    addMessage(
      `<span style="color:red;">Failed to process CSV: ${data.message}</span>`,
    );
  }
}

const runPairTestBtn = document.getElementById("run-pair-test-btn");

const columnList = document.getElementById("column-list");
const addColumnBtn = document.getElementById("add-column-btn");
const deleteColumnBtn = document.getElementById("delete-column-btn");

function switchToPairsTab() {
  document.getElementById("pairs-tab").click();
}

function switchToColumnsTab() {
  document.getElementById("columns-tab").click();
}

function updateColumnList(data) {
  runStats(data);
  populateColumnSelectors(columnData.all);

  columnList.innerHTML = "";

  columnData.all.forEach((column) => {
    const item = document.createElement("div");
    item.className = "column-item list-group-item list-group-item-action";
    item.dataset.column = column;
    item.textContent = column;

    if (columnData.numeric.includes(column)) {
      item.innerHTML += ' <span class="badge bg-info text-dark">numeric</span>';
    } else if (columnData.categorical.includes(column)) {
      item.innerHTML += ' <span class="badge bg-secondary">categorical</span>';
    }

    item.addEventListener("click", function () {
      const previousSelected = columnList.querySelector(".selected-column");
      if (previousSelected) {
        previousSelected.classList.remove("selected-column");
      }

      this.classList.add("selected-column");
      selectedColumn = this.dataset.column;

      deleteColumnBtn.disabled = false;
    });

    columnList.appendChild(item);
  });

}

function handleColumnAdded(data) {
  if (data.success) {
    previewContent.innerHTML = data.preview;

    columnData.all.push(data.column_name);
    if (
      data.column_stats.type.startsWith("int") ||
      data.column_stats.type.startsWith("float")
    ) {
      columnData.numeric.push(data.column_name);
    } else if (data.column_stats.type === "object") {
      columnData.categorical.push(data.column_name);
    }

    updateColumnList(data);

    bootstrap.Modal.getInstance(document.getElementById("columnModal")).hide();

    addMessage(`<b>${data.column_name}</b> added successfully.`);

    switchToColumnsTab();

    enableDownloadButton();
  }
}

function handleColumnModified(data) {
  if (data.success) {
    previewContent.innerHTML = data.preview;

    bootstrap.Modal.getInstance(document.getElementById("columnModal")).hide();

    addMessage(`<b>${data.column_name}</b> modified successfully.`);

    switchToColumnsTab();

    enableDownloadButton();
  }
}

function handleColumnDeleted(data) {
  if (data.success) {
    previewContent.innerHTML = data.preview;

    const index = columnData.all.indexOf(data.column_name);
    if (index > -1) {
      columnData.all.splice(index, 1);
    }

    const numIndex = columnData.numeric.indexOf(data.column_name);
    if (numIndex > -1) {
      columnData.numeric.splice(numIndex, 1);
    }

    const catIndex = columnData.categorical.indexOf(data.column_name);
    if (catIndex > -1) {
      columnData.categorical.splice(catIndex, 1);
    }

    updateColumnList(data);

    selectedColumn = null;
    deleteColumnBtn.disabled = true;

    addMessage(`<b>${data.column_name}</b> deleted successfully.`);

    switchToColumnsTab();

    enableDownloadButton();
  }
}

function handlePairTest(data) {


  if (data.success) {
    previewContent.innerHTML = data.preview;

    pairsContent = document.getElementById("pairs-content");
    startString = `
            <div class="card">
                <div class="card-header">Pair Test Result</div>
                <div class="card-body text-center py-5">
    `
    pairsContent.innerHTML = startString + `
                  <div class="spinner-border" role="status">
                      <span class="visually-hidden">Loading...</span>
                  </div>
        `;

    newString = startString;

    Object.keys(data.result).forEach((e) => {

      newString += `
                <p>
                  <code>${data.result[e].var_one}</code>&nbsp;&nbsp;
                  ${data.result[e].operator}&nbsp;&nbsp;
                  <code>${data.result[e].var_two}</code>
                </p>
                <p>
                  <b>R²</b>: ${data.result[e].rsq}
                </p>
          `;

      if (e != Object.keys(data.result).at(-1)) {
        newString += "<hr style='margin: 3rem;' />"
      }

    });

    endString = "</div> </div> </div>"

    pairsContent.innerHTML = newString;

    updateColumnList(data);
    addMessage(`<b>Pair Test</b> executed successfully.`);

    switchToPairsTab();
  }
}

function enableDownloadButton() {
  const downloadBtn = document.getElementById("download-csv-btn");
  if (downloadBtn) {
    downloadBtn.style.display = "inline-block";
    downloadBtn.href = `/download/${fileId}.csv/`;
  }
}

addColumnBtn.addEventListener("click", function () {
  switchToColumnsTab();

  document.getElementById("column-form").reset();
  document.getElementById("columnModalTitle").textContent = "Add New Column";
  document.getElementById("edit-mode").value = "add";
  document.getElementById("column-name").disabled = false;

  const modal = new bootstrap.Modal(document.getElementById("columnModal"));
  modal.show();
  columnModalInstance = modal;
});

function handleQueryResult(data) {
  if (data.success) {
    queryResultsContent.innerHTML = data.result;
    document.getElementById("query-tab").click();
    addMessage(
      `Command executed successfully. Query resulted in ${data.rows} rows.`,
    );
  } else {
    addMessage(`<span style="color:red;">${data.result}</span>`);
  }
}

document.getElementById("id_file").onchange = function () {
  uploadBtn.click();
};

uploadForm.addEventListener("submit", function (e) {
  e.preventDefault();

  const formData = new FormData(this);
  fileError.textContent = "";
  uploadStatus.innerHTML = "";
  showLoader();

  fetch("/upload/", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        fileId = data.file_id;
        data.message = "Done";
        uploadStatus.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
        addMessage("CSV file uploaded successfully. Processing data...");

        socket.send(
          JSON.stringify({
            action: "process_csv",
            file_id: fileId,
          }),
        );
      } else {
        hideLoader();
        if (data.errors.file) {
          fileError.textContent = data.errors.file;
        }
        uploadStatus.innerHTML = `<div class="alert alert-danger">Upload failed</div>`;
      }
    })
    .catch((error) => {
      hideLoader();
      console.error("Error:", error);
      uploadStatus.innerHTML = `<div class="alert alert-danger">Error uploading file</div>`;
    });
});

sendQueryBtn.addEventListener("click", function () {
  const query = queryInput.value.trim();
  if (query && fileId) {
    addMessage(query, true);
    queryInput.value = "";
    showLoader();

    socket.send(
      JSON.stringify({
        action: "query_data",
        file_id: fileId,
        query: query,
      }),
    );
  }
});

queryInput.addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    sendQueryBtn.click();
  }
});

document.querySelectorAll(".badge").forEach((badge) => {
  badge.addEventListener("click", function () {
    queryInput.value = this.textContent;
    queryInput.focus();
  });
});

document.addEventListener("DOMContentLoaded", function () {
  initWebSocket();
  addMessage("Welcome to AVCS. Upload a CSV file to get started.");
});

document.querySelectorAll(".analysis-btn").forEach((button) => {
  button.addEventListener("click", function () {
    const analysisType = this.getAttribute("data-type");

    const predictorsSelect = document.getElementById("predictors-select");
    const responseSelect = document.getElementById("response-select");

    const predictors = Array.from(predictorsSelect.selectedOptions).map(
      (opt) => opt.value,
    );
    const response = responseSelect.value;

    if (
      (analysisType === "regression" || analysisType == "decisiontrees") &&
      !response
    ) {
      addMessage(
        `<span style="color:red;">Error: This type of analysis requires a response variable</span>`,
      );
      return;
    }

    showLoader();
    document.getElementById("analysis-result-container").innerHTML = `
            <div class="card">
                <div class="card-header">Running ${analysisType} Analysis...</div>
                <div class="card-body text-center py-5">
                    <div class="spinner-border" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            </div>
        `;

    socket.send(
      JSON.stringify({
        action: "analyze_data",
        file_id: fileId,
        analysis_type: analysisType,
        predictors: predictors,
        response: response,
      }),
    );
  });
});

function handleAnalysisResult(data) {
  hideLoader();

  if (!data.success) {
    addMessage(
      `<span style="color:red;">Analysis error: ${data.message}</span>`,
    );
    return;
  }

  const analysisType = data.analysis_type;
  const result = data.result;
  const prompt_response = data.prompt_response;
  const plots = data.plots || {};

  let resultsHtml = `
        <div class="card">
            <div class="card-header">${getAnalysisTitle(analysisType)} Results</div>
            <div class="card-body">
    `;

  if (analysisType != "decisiontrees") {
    Object.keys(plots).forEach((plotKey) => {
      resultsHtml += `
              <div class="mb-4">
                  <h5>${formatPlotTitle(plotKey)}</h5>
                  ${plots[plotKey]}
              </div>
          `;
    });
  } else {
    resultsHtml += `
            <div class="col-12 mb-3">
                <!--<div class="card">-->
                    <!--<div class="card-header">Decision Tree</div>-->
                    <!--<div class="card-body text-center">-->
                        <img src="data:image/png;base64,${plots.tree}" class="img-fluid" alt="Decision Tree">
                    <!--</div>-->
                <!--</div>-->
            </div>
        `;

    resultsHtml += `
            <div class="col-12 mb-3">
                <!--<div class="card">-->
                    <!--<div class="card-header">Importance</div>-->
                    <!--<div class="card-body text-center">-->
                        <img src="data:image/png;base64,${plots.importance}" class="img-fluid" alt="Importance">
                    <!--</div>-->
                <!--</div>-->
            </div>
        `;

    if (plots.predictions) {
      resultsHtml += `
            <div class="col-12 mb-3">
                <!--<div class="card">-->
                    <!--<div class="card-header">Predicitons</div>-->
                    <!--<div class="card-body text-center">-->
                        <img src="data:image/png;base64,${plots.predictions}" class="img-fluid" alt="Predictions">
                <!--    </div>-->
                <!--</div>-->
            </div>
        `;
    }
  }

  function addAnalysisCard(title, content) {
    if (typeof content === "string" && content.includes("</table>")) {
      paddingZero = `style="padding:0;"`;
    } else {
      paddingZero = "";
    }

    htmlResult = `<div class="card mb-3">
            <div class="card-header">${title}</div>
            <div class="card-body" ${paddingZero}>
                    ${content}
                  </div>
              </div>`;

    return htmlResult;
  }

  switch (analysisType) {
    case "summary":
      if (result.summary) {
        resultsHtml += `
                    <h5>Statistical Summary</h5>
                    ${result.summary}
                `;
      }
      if (result.correlation) {
        resultsHtml += `
                    <h5 class="mt-4">Correlation Matrix</h5>
                    ${result.correlation}
                `;
      }
      if (result.missing) {
        resultsHtml += `
                    <h5 class="mt-4">Missing Values</h5>
                    ${result.missing}
                `;
      }
      break;

    case "pca":
      if (result.explained_variance) {
        resultsHtml += `
                    <h5>Explained Variance</h5>
                    ${result.explained_variance}
                `;
      }
      if (result.loadings) {
        resultsHtml += `
                    <h5 class="mt-4">Component Loadings</h5>
                    ${result.loadings}
                `;
      }
      if (result.pca_preview) {
        resultsHtml += `
                    <h5 class="mt-4">PCA Result Preview</h5>
                    ${result.pca_preview}
                `;
      }
      break;

    case "correlation":
      if (result.correlation) {
        resultsHtml += addAnalysisCard(
          "Correlation Matrix",
          result.correlation,
        );
      }
      break;

    case "regression":
      if (result.model_summary) {
        resultsHtml += result.model_summary;
      }
      if (result.coefficients) {
        resultsHtml += addAnalysisCard(
          "Coefficient Estimates",
          result.coefficients,
        );
      }
      break;

    case "clustering":
      if (result.kmeans_summary) {
        resultsHtml += result.kmeans_summary;
      }
      if (result.cluster_centers) {
        resultsHtml += addAnalysisCard(
          "Cluster Centers",
          result.cluster_centers,
        );
      }
      if (result.clustered_data) {
        resultsHtml += addAnalysisCard(
          "Data with Cluster Labels",
          result.clustered_data,
        );
      }
      break;

    case "decisiontrees":
      if (result.model_type == "classification_tree") {
        resultsHtml += addAnalysisCard("Accuracy", result.accuracy);
        resultsHtml += addAnalysisCard(
          "Confusion Matrix",
          result.confusion_matrix,
        );
        let featItem = "";

        Object.keys(result.feature_importance).forEach((feature) => {
          featItem += `
                    <p><strong>${feature}: </strong>${result.feature_importance[feature]}</p>
                  `;
        });

        resultsHtml += addAnalysisCard("Feature Importance", featItem);
      } else if (result.model_type == "regression_tree") {
        let featItem = "";

        Object.keys(result.feature_importance).forEach((feature) => {
          featItem += `
                    <p><strong>${feature}: </strong>${result.feature_importance[feature]}</p>
                  `;
        });

        resultsHtml += addAnalysisCard("Feature Importance", featItem);
      } else {
        resultsHtml += addAnalysisCard("Model Type", Object.keys(result));
      }
      break;

    default:
      resultsHtml += `<p>Analysis completed successfully.</p>`;
  }

  resultsHtml += `
            </div>
        </div>
    `;

  document.getElementById("analysis-result-container").innerHTML = resultsHtml;

  addMessage(
    `<b>${getAnalysisTitle(analysisType)}</b> completed successfully.<br><br>` +
      `${prompt_response}`,
  );
}

function getAnalysisTitle(analysisType) {
  const titles = {
    summary: "Data Summary",
    pca: "Principal Component Analysis",
    correlation: "Correlation Analysis",
    regression: "Linear Regression",
    clustering: "K-Means Clustering",
    decisiontrees: "Decision Tree",
  };
  return titles[analysisType] || "Analysis";
}

function formatPlotTitle(plotKey) {
  return plotKey
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

runPairTestBtn.addEventListener("click", function () {
  const pairResponseSelect = document.getElementById("pair-response-select");
  const response = pairResponseSelect.value;

  if (!response) {
    addMessage(
      `<span style="color:red;">Error: This execution requires a response variable</span>`,
    );
    return;
  }

  showLoader();
  socket.send(
    JSON.stringify({
      action: "test_pairs",
      file_id: fileId,
      response: response,
    }),
  );
});

const saveColumnBtn = document.getElementById("save-column");

saveColumnBtn.addEventListener("click", function () {
  const columnName = document.getElementById("column-name").value.trim();
  const editMode = document.getElementById("edit-mode").value;
  const formula = document.getElementById("column-formula").value.trim();
  const originalColumnName = document.getElementById(
    "original-column-name",
  ).value;

  if (!columnName || !formula) {
    alert("Please fill in all fields");
    return;
  }

  if (editMode === "add" && columnData.all.includes(columnName)) {
    alert(
      `Column "${columnName}" already exists. Please choose a different name.`,
    );
    return;
  }

  showLoader();

  if (editMode === "add") {
    socket.send(
      JSON.stringify({
        action: "add_column",
        file_id: fileId,
        column_name: columnName,
        formula: formula,
      }),
    );
  } else {
    socket.send(
      JSON.stringify({
        action: "modify_column",
        file_id: fileId,
        column_name: originalColumnName || columnName,
        formula: formula,
      }),
    );
  }
});

deleteColumnBtn.addEventListener("click", function () {
  const columnName = columnList
    .querySelector(".selected-column")
    .innerHTML.replace(/ <span.*/, "")
    .trim();

  showLoader();

  socket.send(
    JSON.stringify({
      action: "delete_column",
      file_id: fileId,
      column_name: columnName,
    }),
  );
});



document.getElementById("column-formula").addEventListener("keypress", function(e) {
  const saveColumnBtn = document.getElementById("save-column");
  if (e.key === "Enter") {
    e.preventDefault();
    saveColumnBtn.click();
  }
});
