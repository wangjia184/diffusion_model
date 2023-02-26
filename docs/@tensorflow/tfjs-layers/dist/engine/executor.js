/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/**
 * Executor: Evaluates SymbolicTensor based on feeds.
 */
import { cast, dispose, memory, util } from '@tensorflow/tfjs-core';
import { ValueError } from '../errors';
import { LruCache } from '../utils/executor_utils';
import { toList } from '../utils/generic_utils';
import { InputLayer } from './input_layer';
import { SymbolicTensor } from './topology';
/**
 * Helper function to check the dtype and shape compatibility of a feed value.
 */
function assertFeedCompatibility(key, val) {
    // Check dtype compatibility.
    if (key.dtype == null || key.dtype === val.dtype) {
        //  a.  If types match, return val tensor as is.
        return val;
    }
    try {
        //  b. Attempt to convert to expected type.
        return cast(val, key.dtype);
    }
    catch (err) {
        //  c. If conversion fails, return helpful error.
        throw new ValueError(`The dtype of the feed (${val.dtype}) can not be cast to the dtype ` +
            `of the key '${key.name}' (${key.dtype}).`);
    }
}
/**
 * FeedDict: A mapping from unique SymbolicTensors to feed values for them.
 * A feed value is a concrete value represented as an `Tensor`.
 */
export class FeedDict {
    /**
     * Constructor, optionally does copy-construction.
     * @param feeds An Array of `Feed`s, or another `FeedDict`, in which case
     *   copy-construction will be performed.
     */
    constructor(feeds) {
        this.id2Value = {};
        this.id2Mask = {};
        this.name2Id = {};
        if (feeds instanceof FeedDict) {
            for (const id in feeds.id2Value) {
                this.id2Value[id] = feeds.id2Value[id];
                if (id in feeds.id2Mask) {
                    this.id2Mask[id] = feeds.id2Mask[id];
                }
            }
        }
        else {
            if (feeds == null) {
                return;
            }
            for (const feed of feeds) {
                this.add(feed.key, feed.value);
            }
        }
    }
    /**
     * Add a key-value pair to the FeedDict.
     *
     * @param key The key of the feed.
     * @param value The value of the tensor feed.
     * @param mask The value of the mask feed (optional).
     * @returns This `FeedDict`.
     * @throws ValueError: If the key `SymbolicTensor` already exists in the
     *   `FeedDict`.
     */
    add(key, value, mask) {
        if (this.id2Value[key.id] == null) {
            this.id2Value[key.id] = assertFeedCompatibility(key, value);
            this.name2Id[key.name] = key.id;
            if (mask != null) {
                this.id2Mask[key.id] = mask;
            }
        }
        else {
            throw new ValueError(`Duplicate key: name=${key.name}, id=${key.id}`);
        }
        return this;
    }
    /**
     * Add a Feed to the FeedDict.
     * @param feed The new `Feed` to add.
     * @returns This `FeedDict`.
     */
    addFeed(feed) {
        this.add(feed.key, feed.value);
    }
    /**
     * Probe whether a key already exists in the FeedDict.
     * @param key
     */
    hasKey(key) {
        return this.id2Value[key.id] != null;
    }
    /**
     * Get all the SymbolicTensor available in this FeedDict.
     */
    names() {
        return Object.keys(this.name2Id);
    }
    /**
     * Get the feed value for given key.
     * @param key The SymbolicTensor, or its name (as a string), of which the
     *     value is sought.
     * @returns If `key` exists, the corresponding feed value.
     * @throws ValueError: If `key` does not exist in this `FeedDict`.
     */
    getValue(key) {
        if (key instanceof SymbolicTensor) {
            if (this.id2Value[key.id] == null) {
                throw new ValueError(`Nonexistent key: ${key.name}`);
            }
            else {
                return this.id2Value[key.id];
            }
        }
        else {
            const id = this.name2Id[key];
            if (id == null) {
                throw new ValueError(`Feed dict has no SymbolicTensor name: ${key}`);
            }
            return this.id2Value[id];
        }
    }
    /**
     * Get the feed mask for given key.
     * @param key The SymbolicTensor, or its name (as a string), of which the
     *     value is sought.
     * @returns If `key` exists, the corresponding feed mask.
     * @throws ValueError: If `key` does not exist in this `FeedDict`.
     */
    getMask(key) {
        if (key instanceof SymbolicTensor) {
            if (this.id2Value[key.id] == null) {
                throw new ValueError(`Nonexistent key: ${key.name}`);
            }
            else {
                return this.id2Mask[key.id];
            }
        }
        else {
            const id = this.name2Id[key];
            if (id == null) {
                throw new ValueError(`Feed dict has no SymbolicTensor name: ${key}`);
            }
            return this.id2Mask[id];
        }
    }
    /** Dispose all mask Tensors held by this object. */
    disposeMasks() {
        if (this.id2Mask != null) {
            dispose(this.id2Mask);
        }
    }
}
// Cache for topologically sorted SymbolicTensors for given execution
// targets (i.e., fetches).
export const cachedSorted = new LruCache();
// Cache for recipient count maps for given execution targets (i.e., fetches).
export const cachedRecipientCounts = new LruCache();
export function updateCacheMaxEntries(maxEntries) {
    if (cachedSorted != null) {
        cachedSorted.setMaxEntries(maxEntries);
    }
    if (cachedRecipientCounts != null) {
        cachedRecipientCounts.setMaxEntries(maxEntries);
    }
}
/**
 * Execute a SymbolicTensor by using concrete feed values.
 *
 * A `SymbolicTensor` object is a node in a computation graph of TF.js
 * Layers. The object is backed by a source layer and input
 * `SymbolicTensor`s to the source layer. This method evaluates
 * the `call()` method of the source layer, using concrete values of the
 * inputs obtained from either
 * * `feedDict`, if the input key exists in `feedDict`, or else,
 * * a recursive call to `execute()` itself.
 *
 * @param x: The `SymbolicTensor` to execute.
 * @param feedDict: The feed values, as base condition of the recursion.
 *   execution.
 * @param kwargs: Optional keyword arguments.
 * @param probe: A probe object (of interface `ExecutionProbe`) used for
 *   testing memory footprint of `execute` calls.
 * @returns Result of the execution.
 * @throws ValueError: If any `SymbolicTensor`s from `InputLayer`s
 *   encountered during the execution lacks a feed value in `feedDict`.
 */
export function execute(fetches, feedDict, kwargs, probe) {
    const training = kwargs == null ? false : kwargs['training'];
    const arrayFetches = Array.isArray(fetches);
    const fetchArray = arrayFetches ? fetches : [fetches];
    const outputNames = fetchArray.map(t => t.name);
    const finalOutputs = [];
    const feedNames = feedDict.names();
    for (const outputName of outputNames) {
        if (feedNames.indexOf(outputName) !== -1) {
            finalOutputs.push(feedDict.getValue(outputName));
        }
        else {
            finalOutputs.push(null);
        }
    }
    if (probe != null) {
        // For optional probing of memory footprint during execution.
        probe.maxNumTensors = -Infinity;
        probe.minNumTensors = Infinity;
    }
    // Check cache.
    const fetchAndFeedKey = outputNames.join(',') + '|' + feedDict.names().sort().join(',');
    let sorted = cachedSorted.get(fetchAndFeedKey);
    let recipientCounts;
    if (sorted == null) {
        // Cache doesn't contain the desired combination of fetches. Compute
        // topological sort for the combination for the first time.
        const out = getTopologicalSortAndRecipientCounts(fetchArray, feedDict);
        sorted = out.sorted;
        recipientCounts = out.recipientCounts;
        // Store results in cache for future use.
        cachedSorted.put(fetchAndFeedKey, sorted);
        cachedRecipientCounts.put(fetchAndFeedKey, recipientCounts);
    }
    recipientCounts = {};
    if (!training) {
        Object.assign(recipientCounts, cachedRecipientCounts.get(fetchAndFeedKey));
    }
    const internalFeedDict = new FeedDict(feedDict);
    // Start iterative execution on the topologically-sorted SymbolicTensors.
    for (let i = 0; i < sorted.length; ++i) {
        if (probe != null) {
            // For optional probing of memory usage during execution.
            const numTensors = memory().numTensors;
            if (numTensors > probe.maxNumTensors) {
                probe.maxNumTensors = numTensors;
            }
            if (numTensors < probe.minNumTensors) {
                probe.minNumTensors = numTensors;
            }
        }
        const symbolic = sorted[i];
        const srcLayer = symbolic.sourceLayer;
        if (srcLayer instanceof InputLayer) {
            continue;
        }
        const inputValues = [];
        const inputMasks = [];
        const tensorsToDispose = [];
        let maskExists = false;
        for (const input of symbolic.inputs) {
            const value = internalFeedDict.getValue(input);
            const mask = internalFeedDict.getMask(input);
            inputValues.push(value);
            inputMasks.push(mask);
            if (mask != null) {
                maskExists = true;
            }
            if (!training) {
                recipientCounts[input.name]--;
                if (recipientCounts[input.name] === 0 && !feedDict.hasKey(input) &&
                    outputNames.indexOf(input.name) === -1 && !value.isDisposed &&
                    input.sourceLayer.stateful !== true) {
                    tensorsToDispose.push(value);
                }
            }
        }
        if (maskExists) {
            kwargs = kwargs || {};
            kwargs['mask'] = inputMasks[0];
        }
        const outputTensors = toList(srcLayer.apply(inputValues, kwargs));
        let outputMask = null;
        if (srcLayer.supportsMasking) {
            outputMask = srcLayer.computeMask(inputValues, inputMasks);
        }
        const layerOutputs = getNodeOutputs(symbolic);
        const outputSymbolicTensors = Array.isArray(layerOutputs) ? layerOutputs : [layerOutputs];
        for (let i = 0; i < outputSymbolicTensors.length; ++i) {
            if (!internalFeedDict.hasKey(outputSymbolicTensors[i])) {
                internalFeedDict.add(outputSymbolicTensors[i], outputTensors[i], Array.isArray(outputMask) ? outputMask[0] : outputMask);
            }
            const index = outputNames.indexOf(outputSymbolicTensors[i].name);
            if (index !== -1) {
                finalOutputs[index] = outputTensors[i];
            }
        }
        if (!training) {
            // Clean up Tensors that are no longer needed.
            dispose(tensorsToDispose);
        }
    }
    // NOTE(cais): Unlike intermediate tensors, we don't discard mask
    // tensors as we go, because these tensors are sometimes passed over a
    // series of mutliple layers, i.e., not obeying the immediate input
    // relations in the graph. If this becomes a memory-usage concern,
    // we can improve this in the future.
    internalFeedDict.disposeMasks();
    return arrayFetches ? finalOutputs : finalOutputs[0];
}
/**
 * Sort the `SymbolicTensor`s topologically, for an array of fetches.
 *
 * This function calls getTopologicalSortAndRecipientCountsForOneFetch and
 * merges their results.
 *
 * @param fetch The array of fetches requested. Must be a non-empty array.
 * @param feedDict The dictionary of fed values.
 * @returns sorted: Topologically-sorted array of SymbolicTensors.
 *   recipientCounts: Recipient counts for all SymbolicTensors in `sorted`.
 */
function getTopologicalSortAndRecipientCounts(fetches, feedDict) {
    util.assert(fetches != null && fetches.length > 0, () => `Expected at least one fetch, got none`);
    let finalSorted = [];
    let finalRecipientMap = {};
    if (fetches.length === 1) {
        // Special-casing 1 fetch for efficiency.
        const out = getTopologicalSortAndRecipientCountsForOneFetch(fetches[0], feedDict);
        finalSorted = out.sorted;
        finalRecipientMap = out.recipientMap;
    }
    else {
        const visited = new Set();
        for (const fetch of fetches) {
            const { sorted, recipientMap } = getTopologicalSortAndRecipientCountsForOneFetch(fetch, feedDict);
            // Merge sorted SymbolicTensor Arrays.
            for (const symbolicTensor of sorted) {
                if (!visited.has(symbolicTensor.name)) {
                    finalSorted.push(symbolicTensor);
                    visited.add(symbolicTensor.name);
                }
            }
            // Merge recipient maps.
            for (const name in recipientMap) {
                if (finalRecipientMap[name] == null) {
                    finalRecipientMap[name] = new Set();
                }
                recipientMap[name].forEach(recipient => finalRecipientMap[name].add(recipient));
            }
        }
    }
    return {
        sorted: finalSorted,
        recipientCounts: recipientMap2Counts(finalRecipientMap)
    };
}
function recipientMap2Counts(recipientMap) {
    const recipientCounts = {};
    for (const name in recipientMap) {
        recipientCounts[name] = recipientMap[name].size;
    }
    return recipientCounts;
}
/**
 * Sort the `SymbolicTensor`s topologically, for a single fetch.
 *
 * This helper function processes the upstream SymbolicTensors of a single
 * fetch.
 *
 * @param fetch The single fetch requested.
 * @param feedDict The dictionary of fed values.
 * @returns sorted: Topologically-sorted array of SymbolicTensors.
 *   recipientMap: Recipient names for all SymbolicTensors in `sorted`.
 */
export function getTopologicalSortAndRecipientCountsForOneFetch(fetch, feedDict) {
    const visited = new Set();
    const sorted = [];
    const recipientMap = {};
    // Put keys of the feedDict into visited first, so they don't have to be
    // walked. This is needed in case where there are feeds for intermediate
    // SymbolicTensors of the graph.
    for (const key of feedDict.names()) {
        visited.add(key);
    }
    const stack = [];
    const marks = [];
    // Initial population of stack and marks.
    stack.push(fetch);
    while (stack.length > 0) {
        const top = stack[stack.length - 1];
        if (visited.has(top.name)) {
            stack.pop();
            continue;
        }
        const topIsMarked = marks[marks.length - 1] === stack.length - 1;
        if (top.inputs.length === 0 || topIsMarked) {
            // Input SymbolicTensor or all children have been visited.
            stack.pop();
            sorted.push(top);
            visited.add(top.name);
            if (topIsMarked) {
                marks.pop();
            }
        }
        else {
            // A non-input SymbolicTensor whose upstream SymbolicTensors haven't
            // been visited yet. Push them onto the stack.
            marks.push(stack.length - 1);
            for (const input of top.inputs) {
                // Increment the recipient count. Note that this needs to happen
                // regardless of whether the SymbolicTensor has been visited before.
                if (recipientMap[input.name] == null) {
                    recipientMap[input.name] = new Set();
                }
                recipientMap[input.name].add(top.name);
                if (visited.has(input.name)) {
                    continue; // Avoid repeated visits to the same SymbolicTensor.
                }
                stack.push(input);
            }
        }
    }
    return { sorted, recipientMap };
}
/**
 * Get the symbolic output tensors of the node to which a given fetch belongs.
 * @param fetch The fetched symbolic tensor.
 * @returns The Array of symbolic tensors output by the node to which `fetch`
 *   belongs.
 */
function getNodeOutputs(fetch) {
    let layerOutputs;
    if (fetch.sourceLayer.inboundNodes.length === 1) {
        layerOutputs = fetch.sourceLayer.output;
    }
    else {
        let nodeIndex = null;
        for (let i = 0; i < fetch.sourceLayer.inboundNodes.length; ++i) {
            for (const outputTensor of fetch.sourceLayer.inboundNodes[i]
                .outputTensors) {
                if (outputTensor.id === fetch.id) {
                    nodeIndex = i;
                    break;
                }
            }
        }
        layerOutputs = fetch.sourceLayer.getOutputAt(nodeIndex);
    }
    return layerOutputs;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZXhlY3V0b3IuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvZW5naW5lL2V4ZWN1dG9yLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUg7O0dBRUc7QUFFSCxPQUFPLEVBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxNQUFNLEVBQVUsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFMUUsT0FBTyxFQUFDLFVBQVUsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUVyQyxPQUFPLEVBQUMsUUFBUSxFQUFDLE1BQU0seUJBQXlCLENBQUM7QUFDakQsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBRTlDLE9BQU8sRUFBQyxVQUFVLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFDekMsT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUUxQzs7R0FFRztBQUNILFNBQVMsdUJBQXVCLENBQUMsR0FBbUIsRUFBRSxHQUFXO0lBQy9ELDZCQUE2QjtJQUM3QixJQUFJLEdBQUcsQ0FBQyxLQUFLLElBQUksSUFBSSxJQUFJLEdBQUcsQ0FBQyxLQUFLLEtBQUssR0FBRyxDQUFDLEtBQUssRUFBRTtRQUNoRCxnREFBZ0Q7UUFDaEQsT0FBTyxHQUFHLENBQUM7S0FDWjtJQUNELElBQUk7UUFDRiwyQ0FBMkM7UUFDM0MsT0FBTyxJQUFJLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQztLQUM3QjtJQUFDLE9BQU8sR0FBRyxFQUFFO1FBQ1osaURBQWlEO1FBQ2pELE1BQU0sSUFBSSxVQUFVLENBQ2hCLDBCQUEwQixHQUFHLENBQUMsS0FBSyxpQ0FBaUM7WUFDcEUsZUFBZSxHQUFHLENBQUMsSUFBSSxNQUFNLEdBQUcsQ0FBQyxLQUFLLElBQUksQ0FBQyxDQUFDO0tBQ2pEO0FBQ0gsQ0FBQztBQVVEOzs7R0FHRztBQUNILE1B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RUFBQyxDQUFDO0FBQ2hDLENBQUM7QUFFRDs7Ozs7R0FLRztBQUNILFNBQVMsY0FBYyxDQUFDLEtBQXFCO0lBRTNDLElBQUksWUFBNkMsQ0FBQztJQUNsRCxJQUFJLEtBQUssQ0FBQyxXQUFXLENBQUMsWUFBWSxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQUU7UUFDL0MsWUFBWSxHQUFHLEtBQUssQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDO0tBQ3pDO1NBQU07UUFDTCxJQUFJLFNBQVMsR0FBVyxJQUFJLENBQUM7UUFDN0IsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEtBQUssQ0FBQyxXQUFXLENBQUMsWUFBWSxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtZQUM5RCxLQUFLLE1BQU0sWUFBWSxJQUFJLEtBQUssQ0FBQyxXQUFXLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQztpQkFDbEQsYUFBYSxFQUFFO2dCQUN2QixJQUFJLFlBQVksQ0FBQyxFQUFFLEtBQUssS0FBSyxDQUFDLEVBQUUsRUFBRTtvQkFDaEMsU0FBUyxHQUFHLENBQUMsQ0FBQztvQkFDZCxNQUFNO2lCQUNQO2FBQ0Y7U0FDRjtRQUNELFlBQVksR0FBRyxLQUFLLENBQUMsV0FBVyxDQUFDLFdBQVcsQ0FBQyxTQUFTLENBQUMsQ0FBQztLQUN6RDtJQUNELE9BQU8sWUFBWSxDQUFDO0FBQ3RCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlXG4gKiBsaWNlbnNlIHRoYXQgY2FuIGJlIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgb3IgYXRcbiAqIGh0dHBzOi8vb3BlbnNvdXJjZS5vcmcvbGljZW5zZXMvTUlULlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG4vKipcbiAqIEV4ZWN1dG9yOiBFdmFsdWF0ZXMgU3ltYm9saWNUZW5zb3IgYmFzZWQgb24gZmVlZHMuXG4gKi9cblxuaW1wb3J0IHtjYXN0LCBkaXNwb3NlLCBtZW1vcnksIFRlbnNvciwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtWYWx1ZUVycm9yfSBmcm9tICcuLi9lcnJvcnMnO1xuaW1wb3J0IHtLd2FyZ3N9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7THJ1Q2FjaGV9IGZyb20gJy4uL3V0aWxzL2V4ZWN1dG9yX3V0aWxzJztcbmltcG9ydCB7dG9MaXN0fSBmcm9tICcuLi91dGlscy9nZW5lcmljX3V0aWxzJztcblxuaW1wb3J0IHtJbnB1dExheWVyfSBmcm9tICcuL2lucHV0X2xheWVyJztcbmltcG9ydCB7U3ltYm9saWNUZW5zb3J9IGZyb20gJy4vdG9wb2xvZ3knO1xuXG4vKipcbiAqIEhlbHBlciBmdW5jdGlvbiB0byBjaGVjayB0aGUgZHR5cGUgYW5kIHNoYXBlIGNvbXBhdGliaWxpdHkgb2YgYSBmZWVkIHZhbHVlLlxuICovXG5mdW5jdGlvbiBhc3NlcnRGZWVkQ29tcGF0aWJpbGl0eShrZXk6IFN5bWJvbGljVGVuc29yLCB2YWw6IFRlbnNvcik6IFRlbnNvciB7XG4gIC8vIENoZWNrIGR0eXBlIGNvbXBhdGliaWxpdHkuXG4gIGlmIChrZXkuZHR5cGUgPT0gbnVsbCB8fCBrZXkuZHR5cGUgPT09IHZhbC5kdHlwZSkge1xuICAgIC8vICBhLiAgSWYgdHlwZXMgbWF0Y2gsIHJldHVybiB2YWwgdGVuc29yIGFzIGlzLlxuICAgIHJldHVybiB2YWw7XG4gIH1cbiAgdHJ5IHtcbiAgICAvLyAgYi4gQXR0ZW1wdCB0byBjb252ZXJ0IHRvIGV4cGVjdGVkIHR5cGUuXG4gICAgcmV0dXJuIGNhc3QodmFsLCBrZXkuZHR5cGUpO1xuICB9IGNhdGNoIChlcnIpIHtcbiAgICAvLyAgYy4gSWYgY29udmVyc2lvbiBmYWlscywgcmV0dXJuIGhlbHBmdWwgZXJyb3IuXG4gICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgIGBUaGUgZHR5cGUgb2YgdGhlIGZlZWQgKCR7dmFsLmR0eXBlfSkgY2FuIG5vdCBiZSBjYXN0IHRvIHRoZSBkdHlwZSBgICtcbiAgICAgICAgYG9mIHRoZSBrZXkgJyR7a2V5Lm5hbWV9JyAoJHtrZXkuZHR5cGV9KS5gKTtcbiAgfVxufVxuXG4vKipcbiAqIEEgY29uY3JldGUgVGVuc29yIHZhbHVlIGZvciBhIHN5bWJvbGljIHRlbnNvciBhcyB0aGUga2V5LlxuICovXG5leHBvcnQgaW50ZXJmYWNlIEZlZWQge1xuICBrZXk6IFN5bWJvbGljVGVuc29yO1xuICB2YWx1ZTogVGVuc29yO1xufVxuXG4vKipcbiAqIEZlZWREaWN0OiBBIG1hcHBpbmcgZnJvbSB1bmlxdWUgU3ltYm9saWNUZW5zb3JzIHRvIGZlZWQgdmFsdWVzIGZvciB0aGVtLlxuICogQSBmZWVkIHZhbHVlIGlzIGEgY29uY3JldGUgdmFsdWUgcmVwcmVzZW50ZWQgYXMgYW4gYFRlbnNvcmAuXG4gKi9cbmV4cG9ydCBjbGFzcyBGZWVkRGljdCB7XG4gIHByaXZhdGUgaWQyVmFsdWU6IHtbaWQ6IG51bWJlcl06IFRlbnNvcn0gPSB7fTtcbiAgcHJpdmF0ZSBpZDJNYXNrOiB7W2lkOiBudW1iZXJdOiBUZW5zb3J9ID0ge307XG4gIHByaXZhdGUgbmFtZTJJZDoge1tuYW1lOiBzdHJpbmddOiBudW1iZXJ9ID0ge307XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdG9yLCBvcHRpb25hbGx5IGRvZXMgY29weS1jb25zdHJ1Y3Rpb24uXG4gICAqIEBwYXJhbSBmZWVkcyBBbiBBcnJheSBvZiBgRmVlZGBzLCBvciBhbm90aGVyIGBGZWVkRGljdGAsIGluIHdoaWNoIGNhc2VcbiAgICogICBjb3B5LWNvbnN0cnVjdGlvbiB3aWxsIGJlIHBlcmZvcm1lZC5cbiAgICovXG4gIGNvbnN0cnVjdG9yKGZlZWRzPzogRmVlZFtdfEZlZWREaWN0KSB7XG4gICAgaWYgKGZlZWRzIGluc3RhbmNlb2YgRmVlZERpY3QpIHtcbiAgICAgIGZvciAoY29uc3QgaWQgaW4gZmVlZHMuaWQyVmFsdWUpIHtcbiAgICAgICAgdGhpcy5pZDJWYWx1ZVtpZF0gPSBmZWVkcy5pZDJWYWx1ZVtpZF07XG4gICAgICAgIGlmIChpZCBpbiBmZWVkcy5pZDJNYXNrKSB7XG4gICAgICAgICAgdGhpcy5pZDJNYXNrW2lkXSA9IGZlZWRzLmlkMk1hc2tbaWRdO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIGlmIChmZWVkcyA9PSBudWxsKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGZvciAoY29uc3QgZmVlZCBvZiBmZWVkcykge1xuICAgICAgICB0aGlzLmFkZChmZWVkLmtleSwgZmVlZC52YWx1ZSk7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIEFkZCBhIGtleS12YWx1ZSBwYWlyIHRvIHRoZSBGZWVkRGljdC5cbiAgICpcbiAgICogQHBhcmFtIGtleSBUaGUga2V5IG9mIHRoZSBmZWVkLlxuICAgKiBAcGFyYW0gdmFsdWUgVGhlIHZhbHVlIG9mIHRoZSB0ZW5zb3IgZmVlZC5cbiAgICogQHBhcmFtIG1hc2sgVGhlIHZhbHVlIG9mIHRoZSBtYXNrIGZlZWQgKG9wdGlvbmFsKS5cbiAgICogQHJldHVybnMgVGhpcyBgRmVlZERpY3RgLlxuICAgKiBAdGhyb3dzIFZhbHVlRXJyb3I6IElmIHRoZSBrZXkgYFN5bWJvbGljVGVuc29yYCBhbHJlYWR5IGV4aXN0cyBpbiB0aGVcbiAgICogICBgRmVlZERpY3RgLlxuICAgKi9cbiAgYWRkKGtleTogU3ltYm9saWNUZW5zb3IsIHZhbHVlOiBUZW5zb3IsIG1hc2s/OiBUZW5zb3IpOiBGZWVkRGljdCB7XG4gICAgaWYgKHRoaXMuaWQyVmFsdWVba2V5LmlkXSA9PSBudWxsKSB7XG4gICAgICB0aGlzLmlkMlZhbHVlW2tleS5pZF0gPSBhc3NlcnRGZWVkQ29tcGF0aWJpbGl0eShrZXksIHZhbHVlKTtcbiAgICAgIHRoaXMubmFtZTJJZFtrZXkubmFtZV0gPSBrZXkuaWQ7XG4gICAgICBpZiAobWFzayAhPSBudWxsKSB7XG4gICAgICAgIHRoaXMuaWQyTWFza1trZXkuaWRdID0gbWFzaztcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoYER1cGxpY2F0ZSBrZXk6IG5hbWU9JHtrZXkubmFtZX0sIGlkPSR7a2V5LmlkfWApO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcztcbiAgfVxuXG4gIC8qKlxuICAgKiBBZGQgYSBGZWVkIHRvIHRoZSBGZWVkRGljdC5cbiAgICogQHBhcmFtIGZlZWQgVGhlIG5ldyBgRmVlZGAgdG8gYWRkLlxuICAgKiBAcmV0dXJucyBUaGlzIGBGZWVkRGljdGAuXG4gICAqL1xuICBhZGRGZWVkKGZlZWQ6IEZlZWQpIHtcbiAgICB0aGlzLmFkZChmZWVkLmtleSwgZmVlZC52YWx1ZSk7XG4gIH1cblxuICAvKipcbiAgICogUHJvYmUgd2hldGhlciBhIGtleSBhbHJlYWR5IGV4aXN0cyBpbiB0aGUgRmVlZERpY3QuXG4gICAqIEBwYXJhbSBrZXlcbiAgICovXG4gIGhhc0tleShrZXk6IFN5bWJvbGljVGVuc29yKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIHRoaXMuaWQyVmFsdWVba2V5LmlkXSAhPSBudWxsO1xuICB9XG5cbiAgLyoqXG4gICAqIEdldCBhbGwgdGhlIFN5bWJvbGljVGVuc29yIGF2YWlsYWJsZSBpbiB0aGlzIEZlZWREaWN0LlxuICAgKi9cbiAgbmFtZXMoKTogc3RyaW5nW10ge1xuICAgIHJldHVybiBPYmplY3Qua2V5cyh0aGlzLm5hbWUySWQpO1xuICB9XG5cbiAgLyoqXG4gICAqIEdldCB0aGUgZmVlZCB2YWx1ZSBmb3IgZ2l2ZW4ga2V5LlxuICAgKiBAcGFyYW0ga2V5IFRoZSBTeW1ib2xpY1RlbnNvciwgb3IgaXRzIG5hbWUgKGFzIGEgc3RyaW5nKSwgb2Ygd2hpY2ggdGhlXG4gICAqICAgICB2YWx1ZSBpcyBzb3VnaHQuXG4gICAqIEByZXR1cm5zIElmIGBrZXlgIGV4aXN0cywgdGhlIGNvcnJlc3BvbmRpbmcgZmVlZCB2YWx1ZS5cbiAgICogQHRocm93cyBWYWx1ZUVycm9yOiBJZiBga2V5YCBkb2VzIG5vdCBleGlzdCBpbiB0aGlzIGBGZWVkRGljdGAuXG4gICAqL1xuICBnZXRWYWx1ZShrZXk6IFN5bWJvbGljVGVuc29yfHN0cmluZyk6IFRlbnNvciB7XG4gICAgaWYgKGtleSBpbnN0YW5jZW9mIFN5bWJvbGljVGVuc29yKSB7XG4gICAgICBpZiAodGhpcy5pZDJWYWx1ZVtrZXkuaWRdID09IG51bGwpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoYE5vbmV4aXN0ZW50IGtleTogJHtrZXkubmFtZX1gKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiB0aGlzLmlkMlZhbHVlW2tleS5pZF07XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGlkID0gdGhpcy5uYW1lMklkW2tleV07XG4gICAgICBpZiAoaWQgPT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihgRmVlZCBkaWN0IGhhcyBubyBTeW1ib2xpY1RlbnNvciBuYW1lOiAke2tleX1gKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiB0aGlzLmlkMlZhbHVlW2lkXTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogR2V0IHRoZSBmZWVkIG1hc2sgZm9yIGdpdmVuIGtleS5cbiAgICogQHBhcmFtIGtleSBUaGUgU3ltYm9saWNUZW5zb3IsIG9yIGl0cyBuYW1lIChhcyBhIHN0cmluZyksIG9mIHdoaWNoIHRoZVxuICAgKiAgICAgdmFsdWUgaXMgc291Z2h0LlxuICAgKiBAcmV0dXJucyBJZiBga2V5YCBleGlzdHMsIHRoZSBjb3JyZXNwb25kaW5nIGZlZWQgbWFzay5cbiAgICogQHRocm93cyBWYWx1ZUVycm9yOiBJZiBga2V5YCBkb2VzIG5vdCBleGlzdCBpbiB0aGlzIGBGZWVkRGljdGAuXG4gICAqL1xuICBnZXRNYXNrKGtleTogU3ltYm9saWNUZW5zb3J8c3RyaW5nKTogVGVuc29yIHtcbiAgICBpZiAoa2V5IGluc3RhbmNlb2YgU3ltYm9saWNUZW5zb3IpIHtcbiAgICAgIGlmICh0aGlzLmlkMlZhbHVlW2tleS5pZF0gPT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihgTm9uZXhpc3RlbnQga2V5OiAke2tleS5uYW1lfWApO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmV0dXJuIHRoaXMuaWQyTWFza1trZXkuaWRdO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBpZCA9IHRoaXMubmFtZTJJZFtrZXldO1xuICAgICAgaWYgKGlkID09IG51bGwpIHtcbiAgICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoYEZlZWQgZGljdCBoYXMgbm8gU3ltYm9saWNUZW5zb3IgbmFtZTogJHtrZXl9YCk7XG4gICAgICB9XG4gICAgICByZXR1cm4gdGhpcy5pZDJNYXNrW2lkXTtcbiAgICB9XG4gIH1cblxuICAvKiogRGlzcG9zZSBhbGwgbWFzayBUZW5zb3JzIGhlbGQgYnkgdGhpcyBvYmplY3QuICovXG4gIGRpc3Bvc2VNYXNrcygpIHtcbiAgICBpZiAodGhpcy5pZDJNYXNrICE9IG51bGwpIHtcbiAgICAgIGRpc3Bvc2UodGhpcy5pZDJNYXNrKTtcbiAgICB9XG4gIH1cbn1cblxuLy8gQ2FjaGUgZm9yIHRvcG9sb2dpY2FsbHkgc29ydGVkIFN5bWJvbGljVGVuc29ycyBmb3IgZ2l2ZW4gZXhlY3V0aW9uXG4vLyB0YXJnZXRzIChpLmUuLCBmZXRjaGVzKS5cbmV4cG9ydCBjb25zdCBjYWNoZWRTb3J0ZWQ6IExydUNhY2hlPFN5bWJvbGljVGVuc29yW10+ID1cbiAgICBuZXcgTHJ1Q2FjaGU8U3ltYm9saWNUZW5zb3JbXT4oKTtcblxuLy8gQ2FjaGUgZm9yIHJlY2lwaWVudCBjb3VudCBtYXBzIGZvciBnaXZlbiBleGVjdXRpb24gdGFyZ2V0cyAoaS5lLiwgZmV0Y2hlcykuXG5leHBvcnQgY29uc3QgY2FjaGVkUmVjaXBpZW50Q291bnRzOiBMcnVDYWNoZTxSZWNpcGllbnRDb3VudHM+ID1cbiAgICBuZXcgTHJ1Q2FjaGU8UmVjaXBpZW50Q291bnRzPigpO1xuXG5leHBvcnQgZnVuY3Rpb24gdXBkYXRlQ2FjaGVNYXhFbnRyaWVzKG1heEVudHJpZXM6IG51bWJlcikge1xuICBpZiAoY2FjaGVkU29ydGVkICE9IG51bGwpIHtcbiAgICBjYWNoZWRTb3J0ZWQuc2V0TWF4RW50cmllcyhtYXhFbnRyaWVzKTtcbiAgfVxuICBpZiAoY2FjaGVkUmVjaXBpZW50Q291bnRzICE9IG51bGwpIHtcbiAgICBjYWNoZWRSZWNpcGllbnRDb3VudHMuc2V0TWF4RW50cmllcyhtYXhFbnRyaWVzKTtcbiAgfVxufVxuXG4vKipcbiAqIEludGVyZmFjZSBmb3IgdGhlIG9wdGlvbmFsIG9iamVjdCB1c2VkIGZvciBwcm9iaW5nIHRoZSBtZW1vcnlcbiAqIHVzYWdlIGFuZCBvdGhlciBzdGF0aXN0aWNzIGR1cmluZyBleGVjdXRpb24uXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgRXhlY3V0aW9uUHJvYmUge1xuICAvKipcbiAgICogTWF4aW11bSBudW1iZXIgb2YgdGVuc29ycyB0aGF0IGV4aXN0IGR1cmluZyBhbGwgc3RlcHMgb2YgdGhlXG4gICAqIGV4ZWN1dGlvbi4gVGVuc29yIGNvdW50cyBhcmUgbWVhc3VyZWQgYXQgdGhlIGJlZ2lubmluZyBvZiBldmVyeVxuICAgKiBzdGVwLlxuICAgKi9cbiAgbWF4TnVtVGVuc29ycz86IG51bWJlcjtcblxuICAvKipcbiAgICogTWluaW11bSBudW1iZXIgb2YgdGVuc29ycyB0aGF0IGV4aXN0IGR1cmluZyBhbGwgc3RlcHMgb2YgdGhlXG4gICAqIGV4ZWN1dGlvbi4gVGVuc29yIGNvdW50cyBhcmUgbWVhc3VyZWQgYXQgdGhlIGJlZ2lubmluZyBvZiBldmVyeVxuICAgKiBzdGVwLlxuICAgKi9cbiAgbWluTnVtVGVuc29ycz86IG51bWJlcjtcbn1cblxuLyoqXG4gKiBFeGVjdXRlIGEgU3ltYm9saWNUZW5zb3IgYnkgdXNpbmcgY29uY3JldGUgZmVlZCB2YWx1ZXMuXG4gKlxuICogQSBgU3ltYm9saWNUZW5zb3JgIG9iamVjdCBpcyBhIG5vZGUgaW4gYSBjb21wdXRhdGlvbiBncmFwaCBvZiBURi5qc1xuICogTGF5ZXJzLiBUaGUgb2JqZWN0IGlzIGJhY2tlZCBieSBhIHNvdXJjZSBsYXllciBhbmQgaW5wdXRcbiAqIGBTeW1ib2xpY1RlbnNvcmBzIHRvIHRoZSBzb3VyY2UgbGF5ZXIuIFRoaXMgbWV0aG9kIGV2YWx1YXRlc1xuICogdGhlIGBjYWxsKClgIG1ldGhvZCBvZiB0aGUgc291cmNlIGxheWVyLCB1c2luZyBjb25jcmV0ZSB2YWx1ZXMgb2YgdGhlXG4gKiBpbnB1dHMgb2J0YWluZWQgZnJvbSBlaXRoZXJcbiAqICogYGZlZWREaWN0YCwgaWYgdGhlIGlucHV0IGtleSBleGlzdHMgaW4gYGZlZWREaWN0YCwgb3IgZWxzZSxcbiAqICogYSByZWN1cnNpdmUgY2FsbCB0byBgZXhlY3V0ZSgpYCBpdHNlbGYuXG4gKlxuICogQHBhcmFtIHg6IFRoZSBgU3ltYm9saWNUZW5zb3JgIHRvIGV4ZWN1dGUuXG4gKiBAcGFyYW0gZmVlZERpY3Q6IFRoZSBmZWVkIHZhbHVlcywgYXMgYmFzZSBjb25kaXRpb24gb2YgdGhlIHJlY3Vyc2lvbi5cbiAqICAgZXhlY3V0aW9uLlxuICogQHBhcmFtIGt3YXJnczogT3B0aW9uYWwga2V5d29yZCBhcmd1bWVudHMuXG4gKiBAcGFyYW0gcHJvYmU6IEEgcHJvYmUgb2JqZWN0IChvZiBpbnRlcmZhY2UgYEV4ZWN1dGlvblByb2JlYCkgdXNlZCBmb3JcbiAqICAgdGVzdGluZyBtZW1vcnkgZm9vdHByaW50IG9mIGBleGVjdXRlYCBjYWxscy5cbiAqIEByZXR1cm5zIFJlc3VsdCBvZiB0aGUgZXhlY3V0aW9uLlxuICogQHRocm93cyBWYWx1ZUVycm9yOiBJZiBhbnkgYFN5bWJvbGljVGVuc29yYHMgZnJvbSBgSW5wdXRMYXllcmBzXG4gKiAgIGVuY291bnRlcmVkIGR1cmluZyB0aGUgZXhlY3V0aW9uIGxhY2tzIGEgZmVlZCB2YWx1ZSBpbiBgZmVlZERpY3RgLlxuICovXG5leHBvcnQgZnVuY3Rpb24gZXhlY3V0ZShcbiAgICBmZXRjaGVzOiBTeW1ib2xpY1RlbnNvcnxTeW1ib2xpY1RlbnNvcltdLCBmZWVkRGljdDogRmVlZERpY3QsXG4gICAga3dhcmdzPzogS3dhcmdzLCBwcm9iZT86IEV4ZWN1dGlvblByb2JlKTogVGVuc29yfFxuICAgIFRlbnNvcltdfFtUZW5zb3IgfCBUZW5zb3JbXV0ge1xuICBjb25zdCB0cmFpbmluZzogYm9vbGVhbiA9IGt3YXJncyA9PSBudWxsID8gZmFsc2UgOiBrd2FyZ3NbJ3RyYWluaW5nJ107XG5cbiAgY29uc3QgYXJyYXlGZXRjaGVzID0gQXJyYXkuaXNBcnJheShmZXRjaGVzKTtcbiAgY29uc3QgZmV0Y2hBcnJheTogU3ltYm9saWNUZW5zb3JbXSA9XG4gICAgICBhcnJheUZldGNoZXMgPyBmZXRjaGVzIDogW2ZldGNoZXNdO1xuXG4gIGNvbnN0IG91dHB1dE5hbWVzID0gZmV0Y2hBcnJheS5tYXAodCA9PiB0Lm5hbWUpO1xuICBjb25zdCBmaW5hbE91dHB1dHM6IFRlbnNvcltdID0gW107XG4gIGNvbnN0IGZlZWROYW1lcyA9IGZlZWREaWN0Lm5hbWVzKCk7XG4gIGZvciAoY29uc3Qgb3V0cHV0TmFtZSBvZiBvdXRwdXROYW1lcykge1xuICAgIGlmIChmZWVkTmFtZXMuaW5kZXhPZihvdXRwdXROYW1lKSAhPT0gLTEpIHtcbiAgICAgIGZpbmFsT3V0cHV0cy5wdXNoKGZlZWREaWN0LmdldFZhbHVlKG91dHB1dE5hbWUpKTtcbiAgICB9IGVsc2Uge1xuICAgICAgZmluYWxPdXRwdXRzLnB1c2gobnVsbCk7XG4gICAgfVxuICB9XG5cbiAgaWYgKHByb2JlICE9IG51bGwpIHtcbiAgICAvLyBGb3Igb3B0aW9uYWwgcHJvYmluZyBvZiBtZW1vcnkgZm9vdHByaW50IGR1cmluZyBleGVjdXRpb24uXG4gICAgcHJvYmUubWF4TnVtVGVuc29ycyA9IC1JbmZpbml0eTtcbiAgICBwcm9iZS5taW5OdW1UZW5zb3JzID0gSW5maW5pdHk7XG4gIH1cblxuICAvLyBDaGVjayBjYWNoZS5cbiAgY29uc3QgZmV0Y2hBbmRGZWVkS2V5ID1cbiAgICAgIG91dHB1dE5hbWVzLmpvaW4oJywnKSArICd8JyArIGZlZWREaWN0Lm5hbWVzKCkuc29ydCgpLmpvaW4oJywnKTtcbiAgbGV0IHNvcnRlZDogU3ltYm9saWNUZW5zb3JbXSA9IGNhY2hlZFNvcnRlZC5nZXQoZmV0Y2hBbmRGZWVkS2V5KTtcbiAgbGV0IHJlY2lwaWVudENvdW50czoge1tmZXRjaE5hbWU6IHN0cmluZ106IG51bWJlcn07XG4gIGlmIChzb3J0ZWQgPT0gbnVsbCkge1xuICAgIC8vIENhY2hlIGRvZXNuJ3QgY29udGFpbiB0aGUgZGVzaXJlZCBjb21iaW5hdGlvbiBvZiBmZXRjaGVzLiBDb21wdXRlXG4gICAgLy8gdG9wb2xvZ2ljYWwgc29ydCBmb3IgdGhlIGNvbWJpbmF0aW9uIGZvciB0aGUgZmlyc3QgdGltZS5cbiAgICBjb25zdCBvdXQgPSBnZXRUb3BvbG9naWNhbFNvcnRBbmRSZWNpcGllbnRDb3VudHMoZmV0Y2hBcnJheSwgZmVlZERpY3QpO1xuICAgIHNvcnRlZCA9IG91dC5zb3J0ZWQ7XG4gICAgcmVjaXBpZW50Q291bnRzID0gb3V0LnJlY2lwaWVudENvdW50cztcblxuICAgIC8vIFN0b3JlIHJlc3VsdHMgaW4gY2FjaGUgZm9yIGZ1dHVyZSB1c2UuXG4gICAgY2FjaGVkU29ydGVkLnB1dChmZXRjaEFuZEZlZWRLZXksIHNvcnRlZCk7XG4gICAgY2FjaGVkUmVjaXBpZW50Q291bnRzLnB1dChmZXRjaEFuZEZlZWRLZXksIHJlY2lwaWVudENvdW50cyk7XG4gIH1cbiAgcmVjaXBpZW50Q291bnRzID0ge307XG4gIGlmICghdHJhaW5pbmcpIHtcbiAgICBPYmplY3QuYXNzaWduKHJlY2lwaWVudENvdW50cywgY2FjaGVkUmVjaXBpZW50Q291bnRzLmdldChmZXRjaEFuZEZlZWRLZXkpKTtcbiAgfVxuXG4gIGNvbnN0IGludGVybmFsRmVlZERpY3QgPSBuZXcgRmVlZERpY3QoZmVlZERpY3QpO1xuXG4gIC8vIFN0YXJ0IGl0ZXJhdGl2ZSBleGVjdXRpb24gb24gdGhlIHRvcG9sb2dpY2FsbHktc29ydGVkIFN5bWJvbGljVGVuc29ycy5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBzb3J0ZWQubGVuZ3RoOyArK2kpIHtcbiAgICBpZiAocHJvYmUgIT0gbnVsbCkge1xuICAgICAgLy8gRm9yIG9wdGlvbmFsIHByb2Jpbmcgb2YgbWVtb3J5IHVzYWdlIGR1cmluZyBleGVjdXRpb24uXG4gICAgICBjb25zdCBudW1UZW5zb3JzID0gbWVtb3J5KCkubnVtVGVuc29ycztcbiAgICAgIGlmIChudW1UZW5zb3JzID4gcHJvYmUubWF4TnVtVGVuc29ycykge1xuICAgICAgICBwcm9iZS5tYXhOdW1UZW5zb3JzID0gbnVtVGVuc29ycztcbiAgICAgIH1cbiAgICAgIGlmIChudW1UZW5zb3JzIDwgcHJvYmUubWluTnVtVGVuc29ycykge1xuICAgICAgICBwcm9iZS5taW5OdW1UZW5zb3JzID0gbnVtVGVuc29ycztcbiAgICAgIH1cbiAgICB9XG5cbiAgICBjb25zdCBzeW1ib2xpYyA9IHNvcnRlZFtpXTtcbiAgICBjb25zdCBzcmNMYXllciA9IHN5bWJvbGljLnNvdXJjZUxheWVyO1xuICAgIGlmIChzcmNMYXllciBpbnN0YW5jZW9mIElucHV0TGF5ZXIpIHtcbiAgICAgIGNvbnRpbnVlO1xuICAgIH1cbiAgICBjb25zdCBpbnB1dFZhbHVlczogVGVuc29yW10gPSBbXTtcbiAgICBjb25zdCBpbnB1dE1hc2tzOiBUZW5zb3JbXSA9IFtdO1xuICAgIGNvbnN0IHRlbnNvcnNUb0Rpc3Bvc2U6IFRlbnNvcltdID0gW107XG5cbiAgICBsZXQgbWFza0V4aXN0cyA9IGZhbHNlO1xuICAgIGZvciAoY29uc3QgaW5wdXQgb2Ygc3ltYm9saWMuaW5wdXRzKSB7XG4gICAgICBjb25zdCB2YWx1ZSA9IGludGVybmFsRmVlZERpY3QuZ2V0VmFsdWUoaW5wdXQpO1xuICAgICAgY29uc3QgbWFzayA9IGludGVybmFsRmVlZERpY3QuZ2V0TWFzayhpbnB1dCk7XG4gICAgICBpbnB1dFZhbHVlcy5wdXNoKHZhbHVlKTtcbiAgICAgIGlucHV0TWFza3MucHVzaChtYXNrKTtcbiAgICAgIGlmIChtYXNrICE9IG51bGwpIHtcbiAgICAgICAgbWFza0V4aXN0cyA9IHRydWU7XG4gICAgICB9XG4gICAgICBpZiAoIXRyYWluaW5nKSB7XG4gICAgICAgIHJlY2lwaWVudENvdW50c1tpbnB1dC5uYW1lXS0tO1xuICAgICAgICBpZiAocmVjaXBpZW50Q291bnRzW2lucHV0Lm5hbWVdID09PSAwICYmICFmZWVkRGljdC5oYXNLZXkoaW5wdXQpICYmXG4gICAgICAgICAgICBvdXRwdXROYW1lcy5pbmRleE9mKGlucHV0Lm5hbWUpID09PSAtMSAmJiAhdmFsdWUuaXNEaXNwb3NlZCAmJlxuICAgICAgICAgICAgaW5wdXQuc291cmNlTGF5ZXIuc3RhdGVmdWwgIT09IHRydWUpIHtcbiAgICAgICAgICB0ZW5zb3JzVG9EaXNwb3NlLnB1c2godmFsdWUpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgaWYgKG1hc2tFeGlzdHMpIHtcbiAgICAgIGt3YXJncyA9IGt3YXJncyB8fCB7fTtcbiAgICAgIGt3YXJnc1snbWFzayddID0gaW5wdXRNYXNrc1swXTtcbiAgICB9XG4gICAgY29uc3Qgb3V0cHV0VGVuc29ycyA9XG4gICAgICAgIHRvTGlzdChzcmNMYXllci5hcHBseShpbnB1dFZhbHVlcywga3dhcmdzKSkgYXMgVGVuc29yW107XG4gICAgbGV0IG91dHB1dE1hc2s6IFRlbnNvcnxUZW5zb3JbXSA9IG51bGw7XG4gICAgaWYgKHNyY0xheWVyLnN1cHBvcnRzTWFza2luZykge1xuICAgICAgb3V0cHV0TWFzayA9IHNyY0xheWVyLmNvbXB1dGVNYXNrKGlucHV0VmFsdWVzLCBpbnB1dE1hc2tzKTtcbiAgICB9XG4gICAgY29uc3QgbGF5ZXJPdXRwdXRzID0gZ2V0Tm9kZU91dHB1dHMoc3ltYm9saWMpO1xuICAgIGNvbnN0IG91dHB1dFN5bWJvbGljVGVuc29ycyA9XG4gICAgICAgIEFycmF5LmlzQXJyYXkobGF5ZXJPdXRwdXRzKSA/IGxheWVyT3V0cHV0cyA6IFtsYXllck91dHB1dHNdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgb3V0cHV0U3ltYm9saWNUZW5zb3JzLmxlbmd0aDsgKytpKSB7XG4gICAgICBpZiAoIWludGVybmFsRmVlZERpY3QuaGFzS2V5KG91dHB1dFN5bWJvbGljVGVuc29yc1tpXSkpIHtcbiAgICAgICAgaW50ZXJuYWxGZWVkRGljdC5hZGQoXG4gICAgICAgICAgICBvdXRwdXRTeW1ib2xpY1RlbnNvcnNbaV0sIG91dHB1dFRlbnNvcnNbaV0sXG4gICAgICAgICAgICBBcnJheS5pc0FycmF5KG91dHB1dE1hc2spID8gb3V0cHV0TWFza1swXSA6IG91dHB1dE1hc2spO1xuICAgICAgfVxuICAgICAgY29uc3QgaW5kZXggPSBvdXRwdXROYW1lcy5pbmRleE9mKG91dHB1dFN5bWJvbGljVGVuc29yc1tpXS5uYW1lKTtcbiAgICAgIGlmIChpbmRleCAhPT0gLTEpIHtcbiAgICAgICAgZmluYWxPdXRwdXRzW2luZGV4XSA9IG91dHB1dFRlbnNvcnNbaV07XG4gICAgICB9XG4gICAgfVxuXG4gICAgaWYgKCF0cmFpbmluZykge1xuICAgICAgLy8gQ2xlYW4gdXAgVGVuc29ycyB0aGF0IGFyZSBubyBsb25nZXIgbmVlZGVkLlxuICAgICAgZGlzcG9zZSh0ZW5zb3JzVG9EaXNwb3NlKTtcbiAgICB9XG4gIH1cbiAgLy8gTk9URShjYWlzKTogVW5saWtlIGludGVybWVkaWF0ZSB0ZW5zb3JzLCB3ZSBkb24ndCBkaXNjYXJkIG1hc2tcbiAgLy8gdGVuc29ycyBhcyB3ZSBnbywgYmVjYXVzZSB0aGVzZSB0ZW5zb3JzIGFyZSBzb21ldGltZXMgcGFzc2VkIG92ZXIgYVxuICAvLyBzZXJpZXMgb2YgbXV0bGlwbGUgbGF5ZXJzLCBpLmUuLCBub3Qgb2JleWluZyB0aGUgaW1tZWRpYXRlIGlucHV0XG4gIC8vIHJlbGF0aW9ucyBpbiB0aGUgZ3JhcGguIElmIHRoaXMgYmVjb21lcyBhIG1lbW9yeS11c2FnZSBjb25jZXJuLFxuICAvLyB3ZSBjYW4gaW1wcm92ZSB0aGlzIGluIHRoZSBmdXR1cmUuXG4gIGludGVybmFsRmVlZERpY3QuZGlzcG9zZU1hc2tzKCk7XG5cbiAgcmV0dXJuIGFycmF5RmV0Y2hlcyA/IGZpbmFsT3V0cHV0cyA6IGZpbmFsT3V0cHV0c1swXTtcbn1cblxudHlwZSBSZWNpcGllbnRDb3VudHMgPSB7XG4gIFtmZXRjaE5hbWU6IHN0cmluZ106IG51bWJlclxufTtcblxuZXhwb3J0IHR5cGUgUmVjaXBpZW50TWFwID0ge1xuICBbZmV0Y2hOYW1lOiBzdHJpbmddOiBTZXQ8c3RyaW5nPjtcbn07XG5cbi8qKlxuICogU29ydCB0aGUgYFN5bWJvbGljVGVuc29yYHMgdG9wb2xvZ2ljYWxseSwgZm9yIGFuIGFycmF5IG9mIGZldGNoZXMuXG4gKlxuICogVGhpcyBmdW5jdGlvbiBjYWxscyBnZXRUb3BvbG9naWNhbFNvcnRBbmRSZWNpcGllbnRDb3VudHNGb3JPbmVGZXRjaCBhbmRcbiAqIG1lcmdlcyB0aGVpciByZXN1bHRzLlxuICpcbiAqIEBwYXJhbSBmZXRjaCBUaGUgYXJyYXkgb2YgZmV0Y2hlcyByZXF1ZXN0ZWQuIE11c3QgYmUgYSBub24tZW1wdHkgYXJyYXkuXG4gKiBAcGFyYW0gZmVlZERpY3QgVGhlIGRpY3Rpb25hcnkgb2YgZmVkIHZhbHVlcy5cbiAqIEByZXR1cm5zIHNvcnRlZDogVG9wb2xvZ2ljYWxseS1zb3J0ZWQgYXJyYXkgb2YgU3ltYm9saWNUZW5zb3JzLlxuICogICByZWNpcGllbnRDb3VudHM6IFJlY2lwaWVudCBjb3VudHMgZm9yIGFsbCBTeW1ib2xpY1RlbnNvcnMgaW4gYHNvcnRlZGAuXG4gKi9cbmZ1bmN0aW9uIGdldFRvcG9sb2dpY2FsU29ydEFuZFJlY2lwaWVudENvdW50cyhcbiAgICBmZXRjaGVzOiBTeW1ib2xpY1RlbnNvcltdLCBmZWVkRGljdDogRmVlZERpY3QpOlxuICAgIHtzb3J0ZWQ6IFN5bWJvbGljVGVuc29yW10sIHJlY2lwaWVudENvdW50czogUmVjaXBpZW50Q291bnRzfSB7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgZmV0Y2hlcyAhPSBudWxsICYmIGZldGNoZXMubGVuZ3RoID4gMCxcbiAgICAgICgpID0+IGBFeHBlY3RlZCBhdCBsZWFzdCBvbmUgZmV0Y2gsIGdvdCBub25lYCk7XG5cbiAgbGV0IGZpbmFsU29ydGVkOiBTeW1ib2xpY1RlbnNvcltdID0gW107XG4gIGxldCBmaW5hbFJlY2lwaWVudE1hcDogUmVjaXBpZW50TWFwID0ge307XG4gIGlmIChmZXRjaGVzLmxlbmd0aCA9PT0gMSkge1xuICAgIC8vIFNwZWNpYWwtY2FzaW5nIDEgZmV0Y2ggZm9yIGVmZmljaWVuY3kuXG4gICAgY29uc3Qgb3V0ID1cbiAgICAgICAgZ2V0VG9wb2xvZ2ljYWxTb3J0QW5kUmVjaXBpZW50Q291bnRzRm9yT25lRmV0Y2goZmV0Y2hlc1swXSwgZmVlZERpY3QpO1xuICAgIGZpbmFsU29ydGVkID0gb3V0LnNvcnRlZDtcbiAgICBmaW5hbFJlY2lwaWVudE1hcCA9IG91dC5yZWNpcGllbnRNYXA7XG4gIH0gZWxzZSB7XG4gICAgY29uc3QgdmlzaXRlZCA9IG5ldyBTZXQ8c3RyaW5nPigpO1xuICAgIGZvciAoY29uc3QgZmV0Y2ggb2YgZmV0Y2hlcykge1xuICAgICAgY29uc3Qge3NvcnRlZCwgcmVjaXBpZW50TWFwfSA9XG4gICAgICAgICAgZ2V0VG9wb2xvZ2ljYWxTb3J0QW5kUmVjaXBpZW50Q291bnRzRm9yT25lRmV0Y2goZmV0Y2gsIGZlZWREaWN0KTtcblxuICAgICAgLy8gTWVyZ2Ugc29ydGVkIFN5bWJvbGljVGVuc29yIEFycmF5cy5cbiAgICAgIGZvciAoY29uc3Qgc3ltYm9saWNUZW5zb3Igb2Ygc29ydGVkKSB7XG4gICAgICAgIGlmICghdmlzaXRlZC5oYXMoc3ltYm9saWNUZW5zb3IubmFtZSkpIHtcbiAgICAgICAgICBmaW5hbFNvcnRlZC5wdXNoKHN5bWJvbGljVGVuc29yKTtcbiAgICAgICAgICB2aXNpdGVkLmFkZChzeW1ib2xpY1RlbnNvci5uYW1lKTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICAvLyBNZXJnZSByZWNpcGllbnQgbWFwcy5cbiAgICAgIGZvciAoY29uc3QgbmFtZSBpbiByZWNpcGllbnRNYXApIHtcbiAgICAgICAgaWYgKGZpbmFsUmVjaXBpZW50TWFwW25hbWVdID09IG51bGwpIHtcbiAgICAgICAgICBmaW5hbFJlY2lwaWVudE1hcFtuYW1lXSA9IG5ldyBTZXQ8c3RyaW5nPigpO1xuICAgICAgICB9XG4gICAgICAgIHJlY2lwaWVudE1hcFtuYW1lXS5mb3JFYWNoKFxuICAgICAgICAgICAgcmVjaXBpZW50ID0+IGZpbmFsUmVjaXBpZW50TWFwW25hbWVdLmFkZChyZWNpcGllbnQpKTtcbiAgICAgIH1cbiAgICB9XG4gIH1cbiAgcmV0dXJuIHtcbiAgICBzb3J0ZWQ6IGZpbmFsU29ydGVkLFxuICAgIHJlY2lwaWVudENvdW50czogcmVjaXBpZW50TWFwMkNvdW50cyhmaW5hbFJlY2lwaWVudE1hcClcbiAgfTtcbn1cblxuZnVuY3Rpb24gcmVjaXBpZW50TWFwMkNvdW50cyhyZWNpcGllbnRNYXA6IFJlY2lwaWVudE1hcCk6IFJlY2lwaWVudENvdW50cyB7XG4gIGNvbnN0IHJlY2lwaWVudENvdW50czogUmVjaXBpZW50Q291bnRzID0ge307XG4gIGZvciAoY29uc3QgbmFtZSBpbiByZWNpcGllbnRNYXApIHtcbiAgICByZWNpcGllbnRDb3VudHNbbmFtZV0gPSByZWNpcGllbnRNYXBbbmFtZV0uc2l6ZTtcbiAgfVxuICByZXR1cm4gcmVjaXBpZW50Q291bnRzO1xufVxuXG4vKipcbiAqIFNvcnQgdGhlIGBTeW1ib2xpY1RlbnNvcmBzIHRvcG9sb2dpY2FsbHksIGZvciBhIHNpbmdsZSBmZXRjaC5cbiAqXG4gKiBUaGlzIGhlbHBlciBmdW5jdGlvbiBwcm9jZXNzZXMgdGhlIHVwc3RyZWFtIFN5bWJvbGljVGVuc29ycyBvZiBhIHNpbmdsZVxuICogZmV0Y2guXG4gKlxuICogQHBhcmFtIGZldGNoIFRoZSBzaW5nbGUgZmV0Y2ggcmVxdWVzdGVkLlxuICogQHBhcmFtIGZlZWREaWN0IFRoZSBkaWN0aW9uYXJ5IG9mIGZlZCB2YWx1ZXMuXG4gKiBAcmV0dXJucyBzb3J0ZWQ6IFRvcG9sb2dpY2FsbHktc29ydGVkIGFycmF5IG9mIFN5bWJvbGljVGVuc29ycy5cbiAqICAgcmVjaXBpZW50TWFwOiBSZWNpcGllbnQgbmFtZXMgZm9yIGFsbCBTeW1ib2xpY1RlbnNvcnMgaW4gYHNvcnRlZGAuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBnZXRUb3BvbG9naWNhbFNvcnRBbmRSZWNpcGllbnRDb3VudHNGb3JPbmVGZXRjaChcbiAgICBmZXRjaDogU3ltYm9saWNUZW5zb3IsIGZlZWREaWN0OiBGZWVkRGljdCk6XG4gICAge3NvcnRlZDogU3ltYm9saWNUZW5zb3JbXSwgcmVjaXBpZW50TWFwOiBSZWNpcGllbnRNYXB9IHtcbiAgY29uc3QgdmlzaXRlZCA9IG5ldyBTZXQ8c3RyaW5nPigpO1xuICBjb25zdCBzb3J0ZWQ6IFN5bWJvbGljVGVuc29yW10gPSBbXTtcbiAgY29uc3QgcmVjaXBpZW50TWFwOiBSZWNpcGllbnRNYXAgPSB7fTtcblxuICAvLyBQdXQga2V5cyBvZiB0aGUgZmVlZERpY3QgaW50byB2aXNpdGVkIGZpcnN0LCBzbyB0aGV5IGRvbid0IGhhdmUgdG8gYmVcbiAgLy8gd2Fsa2VkLiBUaGlzIGlzIG5lZWRlZCBpbiBjYXNlIHdoZXJlIHRoZXJlIGFyZSBmZWVkcyBmb3IgaW50ZXJtZWRpYXRlXG4gIC8vIFN5bWJvbGljVGVuc29ycyBvZiB0aGUgZ3JhcGguXG4gIGZvciAoY29uc3Qga2V5IG9mIGZlZWREaWN0Lm5hbWVzKCkpIHtcbiAgICB2aXNpdGVkLmFkZChrZXkpO1xuICB9XG5cbiAgY29uc3Qgc3RhY2s6IFN5bWJvbGljVGVuc29yW10gPSBbXTtcbiAgY29uc3QgbWFya3M6IG51bWJlcltdID0gW107XG5cbiAgLy8gSW5pdGlhbCBwb3B1bGF0aW9uIG9mIHN0YWNrIGFuZCBtYXJrcy5cbiAgc3RhY2sucHVzaChmZXRjaCk7XG5cbiAgd2hpbGUgKHN0YWNrLmxlbmd0aCA+IDApIHtcbiAgICBjb25zdCB0b3AgPSBzdGFja1tzdGFjay5sZW5ndGggLSAxXTtcbiAgICBpZiAodmlzaXRlZC5oYXModG9wLm5hbWUpKSB7XG4gICAgICBzdGFjay5wb3AoKTtcbiAgICAgIGNvbnRpbnVlO1xuICAgIH1cbiAgICBjb25zdCB0b3BJc01hcmtlZCA9IG1hcmtzW21hcmtzLmxlbmd0aCAtIDFdID09PSBzdGFjay5sZW5ndGggLSAxO1xuICAgIGlmICh0b3AuaW5wdXRzLmxlbmd0aCA9PT0gMCB8fCB0b3BJc01hcmtlZCkge1xuICAgICAgLy8gSW5wdXQgU3ltYm9saWNUZW5zb3Igb3IgYWxsIGNoaWxkcmVuIGhhdmUgYmVlbiB2aXNpdGVkLlxuICAgICAgc3RhY2sucG9wKCk7XG4gICAgICBzb3J0ZWQucHVzaCh0b3ApO1xuICAgICAgdmlzaXRlZC5hZGQodG9wLm5hbWUpO1xuICAgICAgaWYgKHRvcElzTWFya2VkKSB7XG4gICAgICAgIG1hcmtzLnBvcCgpO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICAvLyBBIG5vbi1pbnB1dCBTeW1ib2xpY1RlbnNvciB3aG9zZSB1cHN0cmVhbSBTeW1ib2xpY1RlbnNvcnMgaGF2ZW4ndFxuICAgICAgLy8gYmVlbiB2aXNpdGVkIHlldC4gUHVzaCB0aGVtIG9udG8gdGhlIHN0YWNrLlxuICAgICAgbWFya3MucHVzaChzdGFjay5sZW5ndGggLSAxKTtcbiAgICAgIGZvciAoY29uc3QgaW5wdXQgb2YgdG9wLmlucHV0cykge1xuICAgICAgICAvLyBJbmNyZW1lbnQgdGhlIHJlY2lwaWVudCBjb3VudC4gTm90ZSB0aGF0IHRoaXMgbmVlZHMgdG8gaGFwcGVuXG4gICAgICAgIC8vIHJlZ2FyZGxlc3Mgb2Ygd2hldGhlciB0aGUgU3ltYm9saWNUZW5zb3IgaGFzIGJlZW4gdmlzaXRlZCBiZWZvcmUuXG4gICAgICAgIGlmIChyZWNpcGllbnRNYXBbaW5wdXQubmFtZV0gPT0gbnVsbCkge1xuICAgICAgICAgIHJlY2lwaWVudE1hcFtpbnB1dC5uYW1lXSA9IG5ldyBTZXQ8c3RyaW5nPigpO1xuICAgICAgICB9XG4gICAgICAgIHJlY2lwaWVudE1hcFtpbnB1dC5uYW1lXS5hZGQodG9wLm5hbWUpO1xuXG4gICAgICAgIGlmICh2aXNpdGVkLmhhcyhpbnB1dC5uYW1lKSkge1xuICAgICAgICAgIGNvbnRpbnVlOyAgLy8gQXZvaWQgcmVwZWF0ZWQgdmlzaXRzIHRvIHRoZSBzYW1lIFN5bWJvbGljVGVuc29yLlxuICAgICAgICB9XG4gICAgICAgIHN0YWNrLnB1c2goaW5wdXQpO1xuICAgICAgfVxuICAgIH1cbiAgfVxuICByZXR1cm4ge3NvcnRlZCwgcmVjaXBpZW50TWFwfTtcbn1cblxuLyoqXG4gKiBHZXQgdGhlIHN5bWJvbGljIG91dHB1dCB0ZW5zb3JzIG9mIHRoZSBub2RlIHRvIHdoaWNoIGEgZ2l2ZW4gZmV0Y2ggYmVsb25ncy5cbiAqIEBwYXJhbSBmZXRjaCBUaGUgZmV0Y2hlZCBzeW1ib2xpYyB0ZW5zb3IuXG4gKiBAcmV0dXJucyBUaGUgQXJyYXkgb2Ygc3ltYm9saWMgdGVuc29ycyBvdXRwdXQgYnkgdGhlIG5vZGUgdG8gd2hpY2ggYGZldGNoYFxuICogICBiZWxvbmdzLlxuICovXG5mdW5jdGlvbiBnZXROb2RlT3V0cHV0cyhmZXRjaDogU3ltYm9saWNUZW5zb3IpOiBTeW1ib2xpY1RlbnNvcnxcbiAgICBTeW1ib2xpY1RlbnNvcltdIHtcbiAgbGV0IGxheWVyT3V0cHV0czogU3ltYm9saWNUZW5zb3J8U3ltYm9saWNUZW5zb3JbXTtcbiAgaWYgKGZldGNoLnNvdXJjZUxheWVyLmluYm91bmROb2Rlcy5sZW5ndGggPT09IDEpIHtcbiAgICBsYXllck91dHB1dHMgPSBmZXRjaC5zb3VyY2VMYXllci5vdXRwdXQ7XG4gIH0gZWxzZSB7XG4gICAgbGV0IG5vZGVJbmRleDogbnVtYmVyID0gbnVsbDtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGZldGNoLnNvdXJjZUxheWVyLmluYm91bmROb2Rlcy5sZW5ndGg7ICsraSkge1xuICAgICAgZm9yIChjb25zdCBvdXRwdXRUZW5zb3Igb2YgZmV0Y2guc291cmNlTGF5ZXIuaW5ib3VuZE5vZGVzW2ldXG4gICAgICAgICAgICAgICAub3V0cHV0VGVuc29ycykge1xuICAgICAgICBpZiAob3V0cHV0VGVuc29yLmlkID09PSBmZXRjaC5pZCkge1xuICAgICAgICAgIG5vZGVJbmRleCA9IGk7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gICAgbGF5ZXJPdXRwdXRzID0gZmV0Y2guc291cmNlTGF5ZXIuZ2V0T3V0cHV0QXQobm9kZUluZGV4KTtcbiAgfVxuICByZXR1cm4gbGF5ZXJPdXRwdXRzO1xufVxuIl19