document.querySelectorAll(".quiz").forEach((quiz) => {
  const feedback = quiz.querySelector(".feedback");
  quiz.querySelectorAll("button[data-answer]").forEach((button) => {
    button.addEventListener("click", () => {
      const correct = button.dataset.answer === "correct";
      quiz.querySelectorAll("button").forEach((item) => {
        item.classList.remove("correct", "wrong");
      });
      button.classList.add(correct ? "correct" : "wrong");
      feedback.textContent = correct
        ? button.dataset.ok
        : button.dataset.why;
    });
  });
});
