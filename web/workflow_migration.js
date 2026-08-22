import { app } from "/scripts/app.js";

const VIDEO_TRANSPORTS = new Set(["auto", "frames", "video_url"]);

function migrateLoader(values) {
    // v0.3.2: model, projector, thinking, context, batch, micro-batch,
    // GPU layers, free VRAM.
    if (
        values.length >= 8 &&
        ["thinking", "instruct"].includes(values[2]) &&
        typeof values[3] === "number"
    ) {
        return [values[0], values[1], values[4], values[5], values[6], values[7]];
    }
    return values;
}

function migrateChat(values) {
    // v0.3.2 contained max image edge and max image count at positions 11/12.
    // v0.3.3 moves context to Chat and sends every IMAGE batch item unchanged.
    if (
        values.length >= 17 &&
        typeof values[11] === "number" &&
        typeof values[12] === "number" &&
        VIDEO_TRANSPORTS.has(values[14])
    ) {
        return [
            values[0], values[1], values[2], 8192,
            values[3], values[4], values[5], values[6], values[7], values[8],
            values[9], values[10], values[13], values[14], values[15], values[16],
        ];
    }
    return values;
}

function migrateSimpleApi(values) {
    // Both API auth variants ended with max image edge and max image count.
    if (
        values.length >= 11 &&
        typeof values.at(-1) === "number" &&
        typeof values.at(-2) === "number"
    ) {
        return values.slice(0, -2);
    }
    return values;
}

app.registerExtension({
    name: "ComfyUI.MultimodalLLM.WorkflowMigration",
    beforeConfigureGraph(graphData) {
        for (const node of graphData?.nodes ?? []) {
            if (!Array.isArray(node.widgets_values)) continue;
            if (node.type === "MultimodalQwen38Loader") {
                node.widgets_values = migrateLoader(node.widgets_values);
            } else if (node.type === "MultimodalChat") {
                node.widgets_values = migrateChat(node.widgets_values);
            } else if (
                node.type === "MultimodalAPIEnv" ||
                node.type === "MultimodalAPIDirect"
            ) {
                node.widgets_values = migrateSimpleApi(node.widgets_values);
            }
        }
    },
});
