import "./styles.css";
import { App } from "./app";

const root = document.getElementById("app");
if (root) {
  root.setAttribute("aria-busy", "false");
  new App(root);
}
