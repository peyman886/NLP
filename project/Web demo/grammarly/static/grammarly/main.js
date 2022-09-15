$(document).ready(function () {
  $(function () {
    $('[data-toggle="tooltip"]').tooltip();
  });
});

$(document).ready(function () {
  $("#your_text").keyup(function (e) {
    if (e.which == 32) {
      var fd = new FormData();
      fd.append("word", $("#your_text").val());

      $.ajax({
        url: "/predict_word/",
        data: fd,
        processData: false,
        contentType: false,
        type: "POST",
        success: function (data) {
          $("#suggestion").text(data["next_word"]);
          $("#your_text").val(data["normalized_text"]);
        },
        error: function (error) {
          console.log(error);
        },
      });
    }

    if (!$("#suggestion").text().startsWith($("#your_text").val())) {
      $("#suggestion").text("");
    }

    if (e.keyCode == 8) $("#suggestion").text("");
  });
});
