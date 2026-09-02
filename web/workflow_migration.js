import { app } from "/scripts/app.js";

const VIDEO_TRANSPORTS = new Set(["auto", "frames", "video_url"]);
const PROMPT_ORDER_PROPERTY = "qwen38_vl_prompt_order";
const PROMPT_ORDER_NODES = new Set([
    "MultimodalChat",
    "MultimodalAPIEnv",
    "MultimodalAPIDirect",
]);

function markSystemPromptFirst(node) {
    node.properties ??= {};
    node.properties[PROMPT_ORDER_PROPERTY] = "system_first";
}

function swapPromptAndSystemPrompt(values) {
    if (
        values.length < 2 ||
        typeof values[0] !== "string" ||
        typeof values[1] !== "string"
    ) {
        return values;
    }
    return [values[1], values[0], ...values.slice(2)];
}

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
            values[1], values[0], values[2], 8192,
            values[3], values[4], values[5], values[6], values[7], values[8],
            values[9], values[10], values[13], values[14], values[15], values[16],
        ];
    }
    // All other legacy Chat versions only need the first two text widgets
    // exchanged. The marker makes this operation one-time and idempotent.
    return swapPromptAndSystemPrompt(values);
}

function reorderApi(values, systemIndex, hasSeed, hasSeedControl, maxVideoFrames = 8, transport = "frames") {
    const reordered = [
        values[0], values[1], values[2], values[systemIndex], values[3],
        values[4], values[5],
    ];
    if (hasSeed) reordered.push(values[6]);
    else reordered.push(1);
    reordered.push(hasSeedControl ? values[7] : "fixed");
    reordered.push(maxVideoFrames, transport);
    return reordered;
}

function migrateSimpleApi(values) {
    // Current order: base_url, model, key, system_prompt, prompt, max_tokens,
    // temperature, seed, seed-control, max-video-frames, transport.
    if (
        values.length >= 11 &&
        typeof values[3] === "string" &&
        typeof values[4] === "string" &&
        typeof values[5] === "number" &&
        typeof values[6] === "number" &&
        typeof values[7] === "number" &&
        typeof values[8] === "string" &&
        typeof values[9] === "number" &&
        VIDEO_TRANSPORTS.has(values[10])
    ) {
        return values;
    }

    // The last API layout before this change put system_prompt after the
    // required widgets, and already had the video controls.
    if (
        values.length >= 11 &&
        typeof values[3] === "string" &&
        typeof values[4] === "number" &&
        typeof values[5] === "number" &&
        typeof values[6] === "number" &&
        typeof values[7] === "string" &&
        typeof values[8] === "string" &&
        typeof values[9] === "number" &&
        VIDEO_TRANSPORTS.has(values[10])
    ) {
        return reorderApi(values, 8, true, true, values[9], values[10]);
    }

    // Same previous layout on ComfyUI versions that did not serialize the
    // seed-control widget.
    if (
        values.length >= 10 &&
        typeof values[3] === "string" &&
        typeof values[4] === "number" &&
        typeof values[5] === "number" &&
        typeof values[6] === "number" &&
        typeof values[7] === "string" &&
        typeof values[8] === "number" &&
        VIDEO_TRANSPORTS.has(values[9])
    ) {
        return reorderApi(values, 7, true, false, values[8], values[9]);
    }

    // Older API layouts ended with retired max-image controls. Remove them,
    // then put the old system_prompt in its new position.
    let oldValues = values;
    if (
        values.length >= 9 &&
        typeof values.at(-1) === "number" &&
        typeof values.at(-2) === "number"
    ) {
        oldValues = values.slice(0, -2);
    }

    // Before video support: system_prompt was index 8 with seed-control,
    // index 7 with seed but no control widget, or index 6 before seed existed.
    if (
        oldValues.length >= 9 &&
        typeof oldValues[3] === "string" &&
        typeof oldValues[4] === "number" &&
        typeof oldValues[5] === "number" &&
        typeof oldValues[6] === "number" &&
        typeof oldValues[7] === "string" &&
        typeof oldValues[8] === "string"
    ) {
        return reorderApi(oldValues, 8, true, true);
    }
    if (
        oldValues.length >= 8 &&
        typeof oldValues[3] === "string" &&
        typeof oldValues[4] === "number" &&
        typeof oldValues[5] === "number" &&
        typeof oldValues[6] === "number" &&
        typeof oldValues[7] === "string"
    ) {
        return reorderApi(oldValues, 7, true, false);
    }
    if (
        oldValues.length >= 7 &&
        typeof oldValues[3] === "string" &&
        typeof oldValues[4] === "number" &&
        typeof oldValues[5] === "number" &&
        typeof oldValues[6] === "string"
    ) {
        return reorderApi(oldValues, 6, false, false);
    }
    return values;
}

app.registerExtension({
    name: "ComfyUI.MultimodalLLM.WorkflowMigration",
    nodeCreated(node) {
        if (PROMPT_ORDER_NODES.has(node.type)) {
            // Newly created nodes already use the new order; this marker keeps
            // a later graph load from treating them as legacy data.
            markSystemPromptFirst(node);
        }
    },
    beforeConfigureGraph(graphData) {
        for (const node of graphData?.nodes ?? []) {
            if (!Array.isArray(node.widgets_values)) continue;
            if (node.type === "MultimodalQwen38Loader") {
                node.widgets_values = migrateLoader(node.widgets_values);
            } else if (
                node.type === "MultimodalChat" &&
                node.properties?.[PROMPT_ORDER_PROPERTY] !== "system_first"
            ) {
                node.widgets_values = migrateChat(node.widgets_values);
                markSystemPromptFirst(node);
            } else if (
                (node.type === "MultimodalAPIEnv" || node.type === "MultimodalAPIDirect") &&
                node.properties?.[PROMPT_ORDER_PROPERTY] !== "system_first"
            ) {
                node.widgets_values = migrateSimpleApi(node.widgets_values);
                markSystemPromptFirst(node);
            }
        }
    },
});
