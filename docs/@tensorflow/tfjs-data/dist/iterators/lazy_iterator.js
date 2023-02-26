/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs-core';
import * as seedrandom from 'seedrandom';
import { deepClone } from '../util/deep_clone';
import { deepMapAndAwaitAll, deepZip, zipToList } from '../util/deep_map';
import { GrowingRingBuffer } from '../util/growing_ring_buffer';
import { RingBuffer } from '../util/ring_buffer';
// Here we implement a simple asynchronous iterator.
// This lets us avoid using either third-party stream libraries or
// recent TypeScript language support requiring polyfills.
/**
 * Create a `LazyIterator` from an array of items.
 */
export function iteratorFromItems(items) {
    return new ArrayIterator(items);
}
/**
 * Create a `LazyIterator` of incrementing integers.
 */
export function iteratorFromIncrementing(start) {
    let i = start;
    return iteratorFromFunction(() => ({ value: i++, done: false }));
}
/**
 * Create a `LazyIterator` from a function.
 *
 * ```js
 * let i = -1;
 * const func = () =>
 *    ++i < 5 ? {value: i, done: false} : {value: null, done: true};
 * const iter = tf.data.iteratorFromFunction(func);
 * await iter.forEachAsync(e => console.log(e));
 * ```
 *
 * @param func A function that produces data on each call.
 */
export function iteratorFromFunction(func) {
    return new FunctionCallIterator(func);
}
/**
 * Create a `LazyIterator` by concatenating underlying streams, which are
 * themselves provided as a stream.
 *
 * This can also be thought of as a "stream flatten" operation.
 *
 * @param baseIterators A stream of streams to be concatenated.
 * @param baseErrorHandler An optional function that can intercept `Error`s
 *   raised during a `next()` call on the base stream.  This function can decide
 *   whether the error should be propagated, whether the error should be
 *   ignored, or whether the base stream should be terminated.
 */
export function iteratorFromConcatenated(baseIterators, baseErrorHandler) {
    return new ChainedIterator(baseIterators, baseErrorHandler);
}
/**
 * Create a `LazyIterator` by concatenating streams produced by calling a
 * stream-generating function a given number of times.
 *
 * Since a `LazyIterator` is read-once, it cannot be repeated, but this
 * function can be used to achieve a similar effect:
 *
 *   LazyIterator.ofConcatenatedFunction(() => new MyIterator(), 6);
 *
 * @param iteratorFunc: A function that produces a new stream on each call.
 * @param count: The number of times to call the function.
 * @param baseErrorHandler An optional function that can intercept `Error`s
 *   raised during a `next()` call on the base stream.  This function can decide
 *   whether the error should be propagated, whether the error should be
 *   ignored, or whether the base stream should be terminated.
 */
export function iteratorFromConcatenatedFunction(iteratorFunc, count, baseErrorHandler) {
    return iteratorFromConcatenated(iteratorFromFunction(iteratorFunc).take(count), baseErrorHandler);
}
/**
 * Create a `LazyIterator` by zipping together an array, dict, or nested
 * structure of `LazyIterator`s (and perhaps additional constants).
 *
 * The underlying streams must provide elements in a consistent order such
 * that they correspond.
 *
 * Typically, the underlying streams should have the same number of
 * elements. If they do not, the behavior is determined by the
 * `mismatchMode` argument.
 *
 * The nested structure of the `iterators` argument determines the
 * structure of elements in the resulting iterator.
 *
 * @param iterators: An array or object containing LazyIterators at the
 * leaves.
 * @param mismatchMode: Determines what to do when one underlying iterator
 * is exhausted before the others.  `ZipMismatchMode.FAIL` (the default)
 * causes an error to be thrown in this case.  `ZipMismatchMode.SHORTEST`
 * causes the zipped iterator to terminate with the furst underlying
 * streams, so elements remaining on the longer streams are ignored.
 * `ZipMismatchMode.LONGEST` causes the zipped stream to continue, filling
 * in nulls for the exhausted streams, until all streams are exhausted.
 */
export function iteratorFromZipped(iterators, mismatchMode = ZipMismatchMode.FAIL) {
    return new ZipIterator(iterators, mismatchMode);
}
/**
 * An asynchronous iterator, providing lazy access to a potentially
 * unbounded stream of elements.
 *
 * Iterator can be obtained from a dataset:
 * `const iter = await dataset.iterator();`
 */
export class LazyIterator {
    /**
     * Collect all remaining elements of a bounded stream into an array.
     * Obviously this will succeed only for small streams that fit in memory.
     * Useful for testing.
     *
     * @returns A Promise for an array of stream elements, which will resolve
     *   when the stream is exhausted.
     */
    async toArray() {
        const result = [];
        let x = await this.next();
        while (!x.done) {
            result.push(x.value);
            x = await this.next();
        }
        return result;
    }
    /**
     * Collect all elements of this dataset into an array with prefetching 100
     * elements. This is useful for testing, because the prefetch changes the
     * order in which the Promises are resolved along the processing pipeline.
     * This may help expose bugs where results are dependent on the order of
     * Promise resolution rather than on the logical order of the stream (i.e.,
     * due to hidden mutable state).
     *
     * @returns A Promise for an array of stream elements, which will resolve
     *   when the stream is exhausted.
     */
    async toArrayForTest() {
        const stream = this.prefetch(100);
        const result = [];
        let x = await stream.next();
        while (!x.done) {
            result.push(x.value);
            x = await stream.next();
        }
        return result;
    }
    /**
     * Draw items from the stream until it is exhausted.
     *
     * This can be useful when the stream has side effects but no output.  In
     * that case, calling this function guarantees that the stream will be
     * fully processed.
     */
    async resolveFully() {
        let x = await this.next();
        while (!x.done) {
            x = await this.next();
        }
    }
    /**
     * Draw items from the stream until it is exhausted, or a predicate fails.
     *
     * This can be useful when the stream has side effects but no output.  In
     * that case, calling this function guarantees that the stream will be
     * fully processed.
     */
    async resolveWhile(predicate) {
        let x = await this.next();
        let shouldContinue = predicate(x.value);
        while ((!x.done) && shouldContinue) {
            x = await this.next();
            shouldContinue = predicate(x.value);
        }
    }
    /**
     * Handles errors thrown on this stream using a provided handler function.
     *
     * @param handler A function that handles any `Error` thrown during a `next()`
     *   call and returns true if the stream should continue (dropping the failed
     *   call) or false if the stream should quietly terminate.  If the handler
     *   itself throws (or rethrows) an `Error`, that will be propagated.
     *
     * @returns A `LazyIterator` of elements passed through from upstream,
     *   possibly filtering or terminating on upstream `next()` calls that
     *   throw an `Error`.
     */
    handleErrors(handler) {
        return new ErrorHandlingLazyIterator(this, handler);
    }
    // TODO(soergel): Implement reduce() etc.
    /**
     * Filters this stream according to `predicate`.
     *
     * @param predicate A function mapping a stream element to a boolean or a
     * `Promise` for one.
     *
     * @returns A `LazyIterator` of elements for which the predicate was true.
     */
    filter(predicate) {
        return new FilterIterator(this, predicate);
    }
    /**
     * Maps this stream through a 1-to-1 transform.
     *
     * @param transform A function mapping a stream element to a transformed
     *   element.
     *
     * @returns A `LazyIterator` of transformed elements.
     */
    map(transform) {
        return new MapIterator(this, transform);
    }
    /**
     * Maps this stream through an async 1-to-1 transform.
     *
     * @param transform A function mapping a stream element to a `Promise` for a
     *   transformed stream element.
     *
     * @returns A `LazyIterator` of transformed elements.
     */
    mapAsync(transform) {
        return new AsyncMapIterator(this, transform);
    }
    /**
     * Maps this stream through a 1-to-1 transform, forcing serial execution.
     *
     * @param transform A function mapping a stream element to a transformed
     *   element.
     *
     * @returns A `LazyIterator` of transformed elements.
     */
    serialMapAsync(transform) {
        return new AsyncMapIterator(this, transform).serial();
    }
    /**
     * Maps this stream through a 1-to-many transform.
     *
     * @param transform A function mapping a stream element to an array of
     *   transformed elements.
     *
     * @returns A `DataStream` of transformed elements.
     */
    flatmap(transform) {
        return new FlatmapIterator(this, transform);
    }
    /**
     * Apply a function to every element of the stream.
     *
     * @param f A function to apply to each stream element.
     */
    async forEachAsync(f) {
        return this.map(f).resolveFully();
    }
    /**
     * Apply a function to every element of the stream, forcing serial execution.
     *
     * @param f A function to apply to each stream element.  Should return 'true'
     *   to indicate that the stream should continue, or 'false' to cause it to
     *   terminate.
     */
    async serialForEach(f) {
        return this.serialMapAsync(f).resolveWhile(x => (x === true));
    }
    /**
     * Groups elements into batches, represented as arrays of elements.
     *
     * We can think of the elements of this iterator as 'rows' (even if they are
     * nested structures).  By the same token, consecutive values for a given
     * key within the elements form a 'column'.  This matches the usual sense of
     * 'row' and 'column' when processing tabular data (e.g., parsing a CSV).
     *
     * Thus, "Row-major" means that the resulting batch is simply a collection of
     * rows: `[row1, row2, row3, ...]`.  This is contrast to the column-major
     * form, which is needed for vectorized computation.
     *
     * @param batchSize The number of elements desired per batch.
     * @param smallLastBatch Whether to emit the final batch when it has fewer
     *   than batchSize elements. Default true.
     * @returns A `LazyIterator` of batches of elements, represented as arrays
     *   of the original element type.
     */
    rowMajorBatch(batchSize, smallLastBatch = true) {
        return new RowMajorBatchIterator(this, batchSize, smallLastBatch);
    }
    /**
     * Groups elements into batches, represented in column-major form.
     *
     * We can think of the elements of this iterator as 'rows' (even if they are
     * nested structures).  By the same token, consecutive values for a given
     * key within the elements form a 'column'.  This matches the usual sense of
     * 'row' and 'column' when processing tabular data (e.g., parsing a CSV).
     *
     * Thus, "column-major" means that the resulting batch is a (potentially
     * nested) structure representing the columns.  Each column entry, then,
     * contains a collection of the values found in that column for a range of
     * input elements.  This representation allows for vectorized computation, in
     * contrast to the row-major form.
     *
     * The inputs should all have the same nested structure (i.e., of arrays and
     * dicts).  The result is a single object with the same nested structure,
     * where the leaves are arrays collecting the values of the inputs at that
     * location (or, optionally, the result of a custom function applied to those
     * arrays).
     *
     * @param batchSize The number of elements desired per batch.
     * @param smallLastBatch Whether to emit the final batch when it has fewer
     *   than batchSize elements. Default true.
     * @param zipFn: (optional) A function that expects an array of elements at a
     *   single node of the object tree, and returns a `DeepMapResult`.  The
     *   `DeepMapResult` either provides a result value for that node (i.e.,
     *   representing the subtree), or indicates that the node should be processed
     *   recursively.  The default zipFn recurses as far as possible and places
     *   arrays at the leaves.
     * @returns A `LazyIterator` of batches of elements, represented as an object
     *   with collections at the leaves.
     */
    columnMajorBatch(batchSize, smallLastBatch = true, 
    // tslint:disable-next-line:no-any
    zipFn = zipToList) {
        // First collect the desired number of input elements as a row-major batch.
        const rowBatches = this.rowMajorBatch(batchSize, smallLastBatch);
        // Now 'rotate' or 'pivot' the data, collecting all values from each column
        // in the batch (i.e., for each key within the elements) into an array.
        return rowBatches.map(x => deepZip(x, zipFn));
    }
    /**
     * Concatenate this `LazyIterator` with another.
     *
     * @param iterator A `LazyIterator` to be concatenated onto this one.
     * @param baseErrorHandler An optional function that can intercept `Error`s
     *   raised during a `next()` call on the base stream.  This function can
     *   decide whether the error should be propagated, whether the error should
     *   be ignored, or whether the base stream should be terminated.
     * @returns A `LazyIterator`.
     */
    concatenate(iterator, baseErrorHandler) {
        return new ChainedIterator(iteratorFromItems([this, iterator]), baseErrorHandler);
    }
    /**
     * Limits this stream to return at most `count` items.
     *
     * @param count The maximum number of items to provide from the stream. If
     * a negative or undefined value is given, the entire stream is returned
     *   unaltered.
     */
    take(count) {
        if (count < 0 || count == null) {
            return this;
        }
        return new TakeIterator(this, count);
    }
    /**
     * Skips the first `count` items in this stream.
     *
     * @param count The number of items to skip.  If a negative or undefined
     * value is given, the entire stream is returned unaltered.
     */
    skip(count) {
        if (count < 0 || count == null) {
            return this;
        }
        return new SkipIterator(this, count);
    }
    /**
     * Prefetch the first `bufferSize` items in this stream.
     *
     * Note this prefetches Promises, but makes no guarantees about when those
     * Promises resolve.
     *
     * @param bufferSize: An integer specifying the number of elements to be
     *   prefetched.
     */
    prefetch(bufferSize) {
        return new PrefetchIterator(this, bufferSize);
    }
    // TODO(soergel): deep sharded shuffle, where supported
    /**
     * Randomly shuffles the elements of this stream.
     *
     * @param bufferSize: An integer specifying the number of elements from
     * this stream from which the new stream will sample.
     * @param seed: (Optional.) An integer specifying the random seed that
     * will be used to create the distribution.
     */
    shuffle(windowSize, seed) {
        return new ShuffleIterator(this, windowSize, seed);
    }
    /**
     * Force an iterator to execute serially: each next() call will await the
     * prior one, so that they cannot execute concurrently.
     */
    serial() {
        return new SerialIterator(this);
    }
}
// ============================================================================
// The following private classes serve to implement the chainable methods
// on LazyIterator.  Unfortunately they can't be placed in separate files,
// due to resulting trouble with circular imports.
// ============================================================================
// Iterators that just extend LazyIterator directly
// ============================================================================
class ArrayIterator extends LazyIterator {
    constructor(items) {
        super();
        this.items = items;
        this.trav = 0;
    }
    summary() {
        return `Array of ${this.items.length} items`;
    }
    async next() {
        if (this.trav >= this.items.length) {
            return { value: null, done: true };
        }
        const item = this.items[this.trav];
        this.trav++;
        return { value: deepClone(item), done: false };
    }
}
class FunctionCallIterator extends LazyIterator {
    constructor(nextFn) {
        super();
        this.nextFn = nextFn;
    }
    summary() {
        return `Function call`;
    }
    async next() {
        try {
            return this.nextFn();
        }
        catch (e) {
            // Modify the error message but leave the stack trace intact
            e.message =
                `Error thrown while iterating through a dataset: ${e.message}`;
            throw e;
        }
    }
}
class SerialIterator extends LazyIterator {
    constructor(upstream) {
        super();
        this.upstream = upstream;
        this.lastRead = Promise.resolve({ value: null, done: false });
    }
    summary() {
        return `${this.upstream.summary()} -> Serial`;
    }
    async next() {
        // This sets this.lastRead to a new Promise right away, as opposed to
        // saying `await this.lastRead; this.lastRead = this.serialNext();` which
        // would not work because this.nextRead would be updated only after the
        // promise resolves.
        this.lastRead = this.lastRead.then(() => this.serialNext());
        return this.lastRead;
    }
    async serialNext() {
        return this.upstream.next();
    }
}
class SkipIterator extends LazyIterator {
    constructor(upstream, maxCount) {
        super();
        this.upstream = upstream;
        this.maxCount = maxCount;
        // Local state that should not be clobbered by out-of-order execution.
        this.count = 0;
        this.lastRead = Promise.resolve({ value: null, done: false });
    }
    summary() {
        return `${this.upstream.summary()} -> Skip`;
    }
    async next() {
        // This sets this.lastRead to a new Promise right away, as opposed to
        // saying `await this.lastRead; this.lastRead = this.serialNext();` which
        // would not work because this.nextRead would be updated only after the
        // promise resolves.
        this.lastRead = this.lastRead.then(() => this.serialNext());
        return this.lastRead;
    }
    async serialNext() {
        // TODO(soergel): consider tradeoffs of reading in parallel, eg.
        // collecting next() promises in an Array and then waiting for
        // Promise.all() of those. Benefit: pseudo-parallel execution.  Drawback:
        // maybe delayed GC.
        while (this.count++ < this.maxCount) {
            const skipped = await this.upstream.next();
            // short-circuit if upstream is already empty
            if (skipped.done) {
                return skipped;
            }
            tf.dispose(skipped.value);
        }
        return this.upstream.next();
    }
}
class TakeIterator extends LazyIterator {
    constructor(upstream, maxCount) {
        super();
        this.upstream = upstream;
        this.maxCount = maxCount;
        this.count = 0;
    }
    summary() {
        return `${this.upstream.summary()} -> Take`;
    }
    async next() {
        if (this.count++ >= this.maxCount) {
            return { value: null, done: true };
        }
        return this.upstream.next();
    }
}
// Note this batch just groups items into row-wise element arrays.
// Rotating these to a column-wise representation happens only at the dataset
// level.
class RowMajorBatchIterator extends LazyIterator {
    constructor(upstream, batchSize, enableSmallLastBatch = true) {
        super();
        this.upstream = upstream;
        this.batchSize = batchSize;
        this.enableSmallLastBatch = enableSmallLastBatch;
        this.lastRead = Promise.resolve({ value: null, done: false });
    }
    summary() {
        return `${this.upstream.summary()} -> RowMajorBatch`;
    }
    async next() {
        // This sets this.lastRead to a new Promise right away, as opposed to
        // saying `await this.lastRead; this.lastRead = this.serialNext();` which
        // would not work because this.nextRead would be updated only after the
        // promise resolves.
        this.lastRead = this.lastRead.then(() => this.serialNext());
        return this.lastRead;
    }
    async serialNext() {
        const batch = [];
        while (batch.length < this.batchSize) {
            const item = await this.upstream.next();
            if (item.done) {
                if (this.enableSmallLastBatch && batch.length > 0) {
                    return { value: batch, done: false };
                }
                return { value: null, done: true };
            }
            batch.push(item.value);
        }
        return { value: batch, done: false };
    }
}
class FilterIterator extends LazyIterator {
    constructor(upstream, predicate) {
        super();
        this.upstream = upstream;
        this.predicate = predicate;
        this.lastRead = Promise.resolve({ value: null, done: false });
    }
    summary() {
        return `${this.upstream.summary()} -> Filter`;
    }
    async next() {
        // This sets this.lastRead to a new Promise right away, as opposed to
        // saying `await this.lastRead; this.lastRead = this.serialNext();` which
        // would not work because this.nextRead would be updated only after the
        // promise resolves.
        this.lastRead = this.lastRead.then(() => this.serialNext());
        return this.lastRead;
    }
    async serialNext() {
        while (true) {
            const item = await this.upstream.next();
            if (item.done || this.predicate(item.value)) {
                return item;
            }
            tf.dispose(item.value);
        }
    }
}
class MapIterator extends LazyIterator {
    constructor(upstream, transform) {
        super();
        this.upstream = upstream;
        this.transform = transform;
    }
    summary() {
        return `${this.upstream.summary()} -> Map`;
    }
    async next() {
        const item = await this.upstream.next();
        if (item.done) {
            return { value: null, done: true };
        }
        const inputTensors = tf.tensor_util.getTensorsInContainer(item.value);
        // Careful: the transform may mutate the item in place.
        // That's why we have to remember the input Tensors above, and then
        // below dispose only those that were not passed through to the output.
        // Note too that the transform function is responsible for tidying
        // any intermediate Tensors.  Here we are concerned only about the
        // inputs.
        const mapped = this.transform(item.value);
        const outputTensors = tf.tensor_util.getTensorsInContainer(mapped);
        // TODO(soergel) faster intersection
        // TODO(soergel) move to tf.disposeExcept(in, out)?
        for (const t of inputTensors) {
            if (!tf.tensor_util.isTensorInList(t, outputTensors)) {
                t.dispose();
            }
        }
        return { value: mapped, done: false };
    }
}
class ErrorHandlingLazyIterator extends LazyIterator {
    constructor(upstream, handler) {
        super();
        this.upstream = upstream;
        this.handler = handler;
        this.count = 0;
        this.lastRead = Promise.resolve({ value: null, done: false });
    }
    summary() {
        return `${this.upstream.summary()} -> handleErrors`;
    }
    async next() {
        // This sets this.lastRead to a new Promise right away, as opposed to
        // saying `await this.lastRead; this.lastRead = this.serialNext();` which
        // would not work because this.nextRead would be updated only after the
        // promise resolves.
        this.lastRead = this.lastRead.then(() => this.serialNext());
        return this.lastRead;
    }
    async serialNext() {
        while (true) {
            try {
                return await this.upstream.next();
            }
            catch (e) {
                if (!this.handler(e)) {
                    return { value: null, done: true };
                }
                // If the handler returns true, loop and fetch the next upstream item.
                // If the upstream iterator throws an endless stream of errors, and if
                // the handler says to ignore them, then we loop forever here.  That is
                // the correct behavior-- it's up to the handler to decide when to stop.
            }
        }
    }
}
class AsyncMapIterator extends LazyIterator {
    constructor(upstream, transform) {
        super();
        this.upstream = upstream;
        this.transform = transform;
    }
    summary() {
        return `${this.upstream.summary()} -> AsyncMap`;
    }
    async next() {
        const item = await this.upstream.next();
        if (item.done) {
            return { value: null, done: true };
        }
        const inputTensors = tf.tensor_util.getTensorsInContainer(item.value);
        // Careful: the transform may mutate the item in place.
        // That's why we have to remember the input Tensors above, and then
        // below dispose only those that were not passed through to the output.
        // Note too that the transform function is responsible for tidying
        // any intermediate Tensors.  Here we are concerned only about the
        // inputs.
        const mapped = await this.transform(item.value);
        const outputTensors = tf.tensor_util.getTensorsInContainer(mapped);
        // TODO(soergel) faster intersection
        // TODO(soergel) move to tf.disposeExcept(in, out)?
        for (const t of inputTensors) {
            if (!tf.tensor_util.isTensorInList(t, outputTensors)) {
                t.dispose();
            }
        }
        return { value: mapped, done: false };
    }
}
// Iterators that maintain a queue of pending items
// ============================================================================
/**
 * A base class for transforming streams that operate by maintaining an
 * output queue of elements that are ready to return via next().  This is
 * commonly required when the transformation is 1-to-many:  A call to next()
 * may trigger a call to the underlying stream, which will produce many
 * mapped elements of this stream-- of which we need to return only one, so
 * we have to queue the rest.
 */
export class OneToManyIterator extends LazyIterator {
    constructor() {
        super();
        this.outputQueue = new GrowingRingBuffer();
        this.lastRead = Promise.resolve({ value: null, done: false });
    }
    async next() {
        // This sets this.lastRead to a new Promise right away, as opposed to
        // saying `await this.lastRead; this.lastRead = this.serialNext();` which
        // would not work because this.nextRead would be updated only after the
        // promise resolves.
        this.lastRead = this.lastRead.then(() => this.serialNext());
        return this.lastRead;
    }
    async serialNext() {
        // Fetch so that the queue contains at least one item if possible.
        // If the upstream source is exhausted, AND there are no items left in
        // the output queue, then this stream is also exhausted.
        while (this.outputQueue.length() === 0) {
            // TODO(soergel): consider parallel reads.
            if (!await this.pump()) {
                return { value: null, done: true };
            }
        }
        return { value: this.outputQueue.shift(), done: false };
    }
}
class FlatmapIterator extends OneToManyIterator {
    constructor(upstream, transform) {
        super();
        this.upstream = upstream;
        this.transform = transform;
    }
    summary() {
        return `${this.upstream.summary()} -> Flatmap`;
    }
    async pump() {
        const item = await this.upstream.next();
        if (item.done) {
            return false;
        }
        const inputTensors = tf.tensor_util.getTensorsInContainer(item.value);
        // Careful: the transform may mutate the item in place.
        // that's why we have to remember the input Tensors above, and then
        // below dispose only those that were not passed through to the output.
        // Note too that the transform function is responsible for tidying any
        // intermediate Tensors.  Here we are concerned only about the inputs.
        const mappedArray = this.transform(item.value);
        const outputTensors = tf.tensor_util.getTensorsInContainer(mappedArray);
        this.outputQueue.pushAll(mappedArray);
        // TODO(soergel) faster intersection, and deduplicate outputTensors
        // TODO(soergel) move to tf.disposeExcept(in, out)?
        for (const t of inputTensors) {
            if (!tf.tensor_util.isTensorInList(t, outputTensors)) {
                t.dispose();
            }
        }
        return true;
    }
}
/**
 * Provides a `LazyIterator` that concatenates a stream of underlying
 * streams.
 *
 * Doing this in a concurrency-safe way requires some trickery.  In
 * particular, we want this stream to return the elements from the
 * underlying streams in the correct order according to when next() was
 * called, even if the resulting Promises resolve in a different order.
 */
export class ChainedIterator extends LazyIterator {
    constructor(iterators, baseErrorHandler) {
        super();
        this.baseErrorHandler = baseErrorHandler;
        // Strict Promise execution order:
        // a next() call may not even begin until the previous one completes.
        this.lastRead = null;
        // Local state that should not be clobbered by out-of-order execution.
        this.iterator = null;
        this.moreIterators = iterators;
    }
    summary() {
        const upstreamSummaries = 'TODO: fill in upstream of chained summaries';
        return `${upstreamSummaries} -> Chained`;
    }
    async next() {
        this.lastRead = this.readFromChain(this.lastRead);
        return this.lastRead;
    }
    async readFromChain(lastRead) {
        // Must await on the previous read since the previous read may have advanced
        // the stream of streams, from which we need to read.
        // This is unfortunate since we can't parallelize reads. Which means
        // prefetching of chained streams is a no-op.
        // One solution is to prefetch immediately upstream of this.
        await lastRead;
        if (this.iterator == null) {
            const iteratorResult = await this.moreIterators.next();
            if (iteratorResult.done) {
                // No more streams to stream from.
                return { value: null, done: true };
            }
            this.iterator = iteratorResult.value;
            if (this.baseErrorHandler != null) {
                this.iterator = this.iterator.handleErrors(this.baseErrorHandler);
            }
        }
        const itemResult = await this.iterator.next();
        if (itemResult.done) {
            this.iterator = null;
            return this.readFromChain(lastRead);
        }
        return itemResult;
    }
}
export var ZipMismatchMode;
(function (ZipMismatchMode) {
    ZipMismatchMode[ZipMismatchMode["FAIL"] = 0] = "FAIL";
    ZipMismatchMode[ZipMismatchMode["SHORTEST"] = 1] = "SHORTEST";
    ZipMismatchMode[ZipMismatchMode["LONGEST"] = 2] = "LONGEST"; // use nulls for exhausted streams; use up the longest stream.
})(ZipMismatchMode || (ZipMismatchMode = {}));
/**
 * Provides a `LazyIterator` that zips together an array, dict, or nested
 * structure of `LazyIterator`s (and perhaps additional constants).
 *
 * The underlying streams must provide elements in a consistent order such
 * that they correspond.
 *
 * Typically, the underlying streams should have the same number of
 * elements. If they do not, the behavior is determined by the
 * `mismatchMode` argument.
 *
 * The nested structure of the `iterators` argument determines the
 * structure of elements in the resulting iterator.
 *
 * Doing this in a concurrency-safe way requires some trickery.  In
 * particular, we want this stream to return the elements from the
 * underlying streams in the correct order according to when next() was
 * called, even if the resulting Promises resolve in a different order.
 *
 * @param iterators: An array or object containing LazyIterators at the
 * leaves.
 * @param mismatchMode: Determines what to do when one underlying iterator
 * is exhausted before the others.  `ZipMismatchMode.FAIL` (the default)
 * causes an error to be thrown in this case.  `ZipMismatchMode.SHORTEST`
 * causes the zipped iterator to terminate with the furst underlying
 * streams, so elements remaining on the longer streams are ignored.
 * `ZipMismatchMode.LONGEST` causes the zipped stream to continue, filling
 * in nulls for the exhausted streams, until all streams are exhausted.
 */
class ZipIterator extends LazyIterator {
    constructor(iterators, mismatchMode = ZipMismatchMode.FAIL) {
        super();
        this.iterators = iterators;
        this.mismatchMode = mismatchMode;
        this.count = 0;
        this.currentPromise = null;
    }
    summary() {
        const upstreamSummaries = 'TODO: fill in upstream of zip summaries';
        return `{${upstreamSummaries}} -> Zip`;
    }
    async nextState(afterState) {
        // This chaining ensures that the underlying next() are not even called
        // before the previous ones have resolved.
        await afterState;
        // Collect underlying iterator "done" signals as a side effect in
        // getNext()
        let numIterators = 0;
        let iteratorsDone = 0;
        function getNext(container) {
            if (container instanceof LazyIterator) {
                const result = container.next();
                return {
                    value: result.then(x => {
                        numIterators++;
                        if (x.done) {
                            iteratorsDone++;
                        }
                        return x.value;
                    }),
                    recurse: false
                };
            }
            else {
                return { value: null, recurse: true };
            }
        }
        const mapped = await deepMapAndAwaitAll(this.iterators, getNext);
        if (numIterators === iteratorsDone) {
            // The streams have all ended.
            return { value: null, done: true };
        }
        if (iteratorsDone > 0) {
            switch (this.mismatchMode) {
                case ZipMismatchMode.FAIL:
                    throw new Error('Zipped streams should have the same length. ' +
                        `Mismatched at element ${this.count}.`);
                case ZipMismatchMode.SHORTEST:
                    return { value: null, done: true };
                case ZipMismatchMode.LONGEST:
                default:
                // Continue.  The exhausted streams already produced value: null.
            }
        }
        this.count++;
        return { value: mapped, done: false };
    }
    async next() {
        this.currentPromise = this.nextState(this.currentPromise);
        return this.currentPromise;
    }
}
// Iterators that maintain a ring buffer of pending promises
// ============================================================================
/**
 * A stream that prefetches a given number of items from an upstream source,
 * returning them in FIFO order.
 *
 * Note this prefetches Promises, but makes no guarantees about when those
 * Promises resolve.
 */
export class PrefetchIterator extends LazyIterator {
    constructor(upstream, bufferSize) {
        super();
        this.upstream = upstream;
        this.bufferSize = bufferSize;
        this.buffer = new RingBuffer(bufferSize);
    }
    summary() {
        return `${this.upstream.summary()} -> Prefetch`;
    }
    /**
     * Refill the prefetch buffer.  Returns only after the buffer is full, or
     * the upstream source is exhausted.
     */
    refill() {
        while (!this.buffer.isFull()) {
            const v = this.upstream.next();
            this.buffer.push(v);
        }
    }
    next() {
        this.refill();
        // This shift will never throw an error because the buffer is always
        // full after a refill. If the stream is exhausted, the buffer will be
        // full of Promises that will resolve to the end-of-stream signal.
        return this.buffer.shift();
    }
}
/**
 * A stream that performs a sliding-window random shuffle on an upstream
 * source. This is like a `PrefetchIterator` except that the items are
 * returned in randomized order.  Mixing naturally improves as the buffer
 * size increases.
 */
export class ShuffleIterator extends PrefetchIterator {
    constructor(upstream, windowSize, seed) {
        super(upstream, windowSize);
        this.upstream = upstream;
        this.windowSize = windowSize;
        // Local state that should not be clobbered by out-of-order execution.
        this.upstreamExhausted = false;
        this.random = seedrandom.alea(seed || tf.util.now().toString());
        this.lastRead = Promise.resolve({ value: null, done: false });
    }
    async next() {
        // This sets this.lastRead to a new Promise right away, as opposed to
        // saying `await this.lastRead; this.lastRead = this.serialNext();` which
        // would not work because this.nextRead would be updated only after the
        // promise resolves.
        this.lastRead = this.lastRead.then(() => this.serialNext());
        return this.lastRead;
    }
    randomInt(max) {
        return Math.floor(this.random() * max);
    }
    chooseIndex() {
        return this.randomInt(this.buffer.length());
    }
    async serialNext() {
        // TODO(soergel): consider performance
        if (!this.upstreamExhausted) {
            this.refill();
        }
        while (!this.buffer.isEmpty()) {
            const chosenIndex = this.chooseIndex();
            const result = await this.buffer.shuffleExcise(chosenIndex);
            if (result.done) {
                this.upstreamExhausted = true;
            }
            else {
                this.refill();
                return result;
            }
        }
        return { value: null, done: true };
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibGF6eV9pdGVyYXRvci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtZGF0YS9zcmMvaXRlcmF0b3JzL2xhenlfaXRlcmF0b3IudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7Ozs7R0FnQkc7QUFFSCxPQUFPLEtBQUssRUFBRSxNQUFNLHVCQUF1QixDQUFDO0FBQzVDLE9BQU8sS0FBSyxVQUFVLE1BQU0sWUFBWSxDQUFDO0FBR3pDLE9BQU8sRUFBQyxTQUFTLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUM3QyxPQUFPLEVBQUMsa0JBQWtCLEVBQXFDLE9BQU8sRUFBRSxTQUFTLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUMzRyxPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSw2QkFBNkIsQ0FBQztBQUM5RCxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFPL0Msb0RBQW9EO0FBQ3BELGtFQUFrRTtBQUNsRSwwREFBMEQ7QUFFMUQ7O0dBRUc7QUFDSCxNQUFNLFVBQVUsaUJBQWlCLENBQUksS0FBVTtJQUM3QyxPQUFPLElBQUksYUFBYSxDQUFDLEtBQUssQ0FBQyxDQUFDO0FBQ2xDLENBQUM7QUFFRDs7R0FFRztBQUNILE1BQU0sVUFBVSx3QkFBd0IsQ0FBQyxLQUFhO0lBQ3BELElBQUksQ0FBQyxHQUFHLEtBQUssQ0FBQztJQUNkLE9BQU8sb0JBQW9CLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFDLEtBQUssRUFBRSxDQUFDLEVBQUUsRUFBRSxJQUFJLEVBQUUsS0FBSyxFQUFDLENBQUMsQ0FBQyxDQUFDO0FBQ2pFLENBQUM7QUFFRDs7Ozs7Ozs7Ozs7O0dBWUc7QUFDSCxNQUFNLFVBQVUsb0JBQW9CLENBQ2hDLElBQ2lEO0lBQ25ELE9BQU8sSUFBSSxvQkFBb0IsQ0FBQyxJQUFJLENBQUMsQ0FBQztBQUN4QyxDQUFDO0FBRUQ7Ozs7Ozs7Ozs7O0dB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Q0FBQyxpQkFBaUIsR0FBRyxJQUFJLENBQUM7YUFDL0I7aUJBQU07Z0JBQ0wsSUFBSSxDQUFDLE1BQU0sRUFBRSxDQUFDO2dCQUNkLE9BQU8sTUFBTSxDQUFDO2FBQ2Y7U0FDRjtRQUNELE9BQU8sRUFBQyxLQUFLLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUMsQ0FBQztJQUNuQyxDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQgKiBhcyB0ZiBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0ICogYXMgc2VlZHJhbmRvbSBmcm9tICdzZWVkcmFuZG9tJztcblxuaW1wb3J0IHtDb250YWluZXJ9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7ZGVlcENsb25lfSBmcm9tICcuLi91dGlsL2RlZXBfY2xvbmUnO1xuaW1wb3J0IHtkZWVwTWFwQW5kQXdhaXRBbGwsIERlZXBNYXBBc3luY1Jlc3VsdCwgRGVlcE1hcFJlc3VsdCwgZGVlcFppcCwgemlwVG9MaXN0fSBmcm9tICcuLi91dGlsL2RlZXBfbWFwJztcbmltcG9ydCB7R3Jvd2luZ1JpbmdCdWZmZXJ9IGZyb20gJy4uL3V0aWwvZ3Jvd2luZ19yaW5nX2J1ZmZlcic7XG5pbXBvcnQge1JpbmdCdWZmZXJ9IGZyb20gJy4uL3V0aWwvcmluZ19idWZmZXInO1xuXG4vKipcbiAqIEEgbmVzdGVkIHN0cnVjdHVyZSBvZiBMYXp5SXRlcmF0b3JzLCB1c2VkIGFzIHRoZSBpbnB1dCB0byB6aXAoKS5cbiAqL1xuZXhwb3J0IHR5cGUgSXRlcmF0b3JDb250YWluZXIgPSBDb250YWluZXI8TGF6eUl0ZXJhdG9yPHRmLlRlbnNvckNvbnRhaW5lcj4+O1xuXG4vLyBIZXJlIHdlIGltcGxlbWVudCBhIHNpbXBsZSBhc3luY2hyb25vdXMgaXRlcmF0b3IuXG4vLyBUaGlzIGxldHMgdXMgYXZvaWQgdXNpbmcgZWl0aGVyIHRoaXJkLXBhcnR5IHN0cmVhbSBsaWJyYXJpZXMgb3Jcbi8vIHJlY2VudCBUeXBlU2NyaXB0IGxhbmd1YWdlIHN1cHBvcnQgcmVxdWlyaW5nIHBvbHlmaWxscy5cblxuLyoqXG4gKiBDcmVhdGUgYSBgTGF6eUl0ZXJhdG9yYCBmcm9tIGFuIGFycmF5IG9mIGl0ZW1zLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaXRlcmF0b3JGcm9tSXRlbXM8VD4oaXRlbXM6IFRbXSk6IExhenlJdGVyYXRvcjxUPiB7XG4gIHJldHVybiBuZXcgQXJyYXlJdGVyYXRvcihpdGVtcyk7XG59XG5cbi8qKlxuICogQ3JlYXRlIGEgYExhenlJdGVyYXRvcmAgb2YgaW5jcmVtZW50aW5nIGludGVnZXJzLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaXRlcmF0b3JGcm9tSW5jcmVtZW50aW5nKHN0YXJ0OiBudW1iZXIpOiBMYXp5SXRlcmF0b3I8bnVtYmVyPiB7XG4gIGxldCBpID0gc3RhcnQ7XG4gIHJldHVybiBpdGVyYXRvckZyb21GdW5jdGlvbigoKSA9PiAoe3ZhbHVlOiBpKyssIGRvbmU6IGZhbHNlfSkpO1xufVxuXG4vKipcbiAqIENyZWF0ZSBhIGBMYXp5SXRlcmF0b3JgIGZyb20gYSBmdW5jdGlvbi5cbiAqXG4gKiBgYGBqc1xuICogbGV0IGkgPSAtMTtcbiAqIGNvbnN0IGZ1bmMgPSAoKSA9PlxuICogICAgKytpIDwgNSA/IHt2YWx1ZTogaSwgZG9uZTogZmFsc2V9IDoge3ZhbHVlOiBudWxsLCBkb25lOiB0cnVlfTtcbiAqIGNvbnN0IGl0ZXIgPSB0Zi5kYXRhLml0ZXJhdG9yRnJvbUZ1bmN0aW9uKGZ1bmMpO1xuICogYXdhaXQgaXRlci5mb3JFYWNoQXN5bmMoZSA9PiBjb25zb2xlLmxvZyhlKSk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gZnVuYyBBIGZ1bmN0aW9uIHRoYXQgcHJvZHVjZXMgZGF0YSBvbiBlYWNoIGNhbGwuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBpdGVyYXRvckZyb21GdW5jdGlvbjxUPihcbiAgICBmdW5jOiAoKSA9PlxuICAgICAgICBJdGVyYXRvclJlc3VsdDxUPnwgUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxUPj4pOiBMYXp5SXRlcmF0b3I8VD4ge1xuICByZXR1cm4gbmV3IEZ1bmN0aW9uQ2FsbEl0ZXJhdG9yKGZ1bmMpO1xufVxuXG4vKipcbiAqIENyZWF0ZSBhIGBMYXp5SXRlcmF0b3JgIGJ5IGNvbmNhdGVuYXRpbmcgdW5kZXJseWluZyBzdHJlYW1zLCB3aGljaCBhcmVcbiAqIHRoZW1zZWx2ZXMgcHJvdmlkZWQgYXMgYSBzdHJlYW0uXG4gKlxuICogVGhpcyBjYW4gYWxzbyBiZSB0aG91Z2h0IG9mIGFzIGEgXCJzdHJlYW0gZmxhdHRlblwiIG9wZXJhdGlvbi5cbiAqXG4gKiBAcGFyYW0gYmFzZUl0ZXJhdG9ycyBBIHN0cmVhbSBvZiBzdHJlYW1zIHRvIGJlIGNvbmNhdGVuYXRlZC5cbiAqIEBwYXJhbSBiYXNlRXJyb3JIYW5kbGVyIEFuIG9wdGlvbmFsIGZ1bmN0aW9uIHRoYXQgY2FuIGludGVyY2VwdCBgRXJyb3Jgc1xuICogICByYWlzZWQgZHVyaW5nIGEgYG5leHQoKWAgY2FsbCBvbiB0aGUgYmFzZSBzdHJlYW0uICBUaGlzIGZ1bmN0aW9uIGNhbiBkZWNpZGVcbiAqICAgd2hldGhlciB0aGUgZXJyb3Igc2hvdWxkIGJlIHByb3BhZ2F0ZWQsIHdoZXRoZXIgdGhlIGVycm9yIHNob3VsZCBiZVxuICogICBpZ25vcmVkLCBvciB3aGV0aGVyIHRoZSBiYXNlIHN0cmVhbSBzaG91bGQgYmUgdGVybWluYXRlZC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGl0ZXJhdG9yRnJvbUNvbmNhdGVuYXRlZDxUPihcbiAgICBiYXNlSXRlcmF0b3JzOiBMYXp5SXRlcmF0b3I8TGF6eUl0ZXJhdG9yPFQ+PixcbiAgICBiYXNlRXJyb3JIYW5kbGVyPzogKGU6IEVycm9yKSA9PiBib29sZWFuKTogTGF6eUl0ZXJhdG9yPFQ+IHtcbiAgcmV0dXJuIG5ldyBDaGFpbmVkSXRlcmF0b3IoYmFzZUl0ZXJhdG9ycywgYmFzZUVycm9ySGFuZGxlcik7XG59XG5cbi8qKlxuICogQ3JlYXRlIGEgYExhenlJdGVyYXRvcmAgYnkgY29uY2F0ZW5hdGluZyBzdHJlYW1zIHByb2R1Y2VkIGJ5IGNhbGxpbmcgYVxuICogc3RyZWFtLWdlbmVyYXRpbmcgZnVuY3Rpb24gYSBnaXZlbiBudW1iZXIgb2YgdGltZXMuXG4gKlxuICogU2luY2UgYSBgTGF6eUl0ZXJhdG9yYCBpcyByZWFkLW9uY2UsIGl0IGNhbm5vdCBiZSByZXBlYXRlZCwgYnV0IHRoaXNcbiAqIGZ1bmN0aW9uIGNhbiBiZSB1c2VkIHRvIGFjaGlldmUgYSBzaW1pbGFyIGVmZmVjdDpcbiAqXG4gKiAgIExhenlJdGVyYXRvci5vZkNvbmNhdGVuYXRlZEZ1bmN0aW9uKCgpID0+IG5ldyBNeUl0ZXJhdG9yKCksIDYpO1xuICpcbiAqIEBwYXJhbSBpdGVyYXRvckZ1bmM6IEEgZnVuY3Rpb24gdGhhdCBwcm9kdWNlcyBhIG5ldyBzdHJlYW0gb24gZWFjaCBjYWxsLlxuICogQHBhcmFtIGNvdW50OiBUaGUgbnVtYmVyIG9mIHRpbWVzIHRvIGNhbGwgdGhlIGZ1bmN0aW9uLlxuICogQHBhcmFtIGJhc2VFcnJvckhhbmRsZXIgQW4gb3B0aW9uYWwgZnVuY3Rpb24gdGhhdCBjYW4gaW50ZXJjZXB0IGBFcnJvcmBzXG4gKiAgIHJhaXNlZCBkdXJpbmcgYSBgbmV4dCgpYCBjYWxsIG9uIHRoZSBiYXNlIHN0cmVhbS4gIFRoaXMgZnVuY3Rpb24gY2FuIGRlY2lkZVxuICogICB3aGV0aGVyIHRoZSBlcnJvciBzaG91bGQgYmUgcHJvcGFnYXRlZCwgd2hldGhlciB0aGUgZXJyb3Igc2hvdWxkIGJlXG4gKiAgIGlnbm9yZWQsIG9yIHdoZXRoZXIgdGhlIGJhc2Ugc3RyZWFtIHNob3VsZCBiZSB0ZXJtaW5hdGVkLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaXRlcmF0b3JGcm9tQ29uY2F0ZW5hdGVkRnVuY3Rpb248VD4oXG4gICAgaXRlcmF0b3JGdW5jOiAoKSA9PiBJdGVyYXRvclJlc3VsdDxMYXp5SXRlcmF0b3I8VD4+LCBjb3VudDogbnVtYmVyLFxuICAgIGJhc2VFcnJvckhhbmRsZXI/OiAoZTogRXJyb3IpID0+IGJvb2xlYW4pOiBMYXp5SXRlcmF0b3I8VD4ge1xuICByZXR1cm4gaXRlcmF0b3JGcm9tQ29uY2F0ZW5hdGVkKFxuICAgICAgaXRlcmF0b3JGcm9tRnVuY3Rpb24oaXRlcmF0b3JGdW5jKS50YWtlKGNvdW50KSwgYmFzZUVycm9ySGFuZGxlcik7XG59XG5cbi8qKlxuICogQ3JlYXRlIGEgYExhenlJdGVyYXRvcmAgYnkgemlwcGluZyB0b2dldGhlciBhbiBhcnJheSwgZGljdCwgb3IgbmVzdGVkXG4gKiBzdHJ1Y3R1cmUgb2YgYExhenlJdGVyYXRvcmBzIChhbmQgcGVyaGFwcyBhZGRpdGlvbmFsIGNvbnN0YW50cykuXG4gKlxuICogVGhlIHVuZGVybHlpbmcgc3RyZWFtcyBtdXN0IHByb3ZpZGUgZWxlbWVudHMgaW4gYSBjb25zaXN0ZW50IG9yZGVyIHN1Y2hcbiAqIHRoYXQgdGhleSBjb3JyZXNwb25kLlxuICpcbiAqIFR5cGljYWxseSwgdGhlIHVuZGVybHlpbmcgc3RyZWFtcyBzaG91bGQgaGF2ZSB0aGUgc2FtZSBudW1iZXIgb2ZcbiAqIGVsZW1lbnRzLiBJZiB0aGV5IGRvIG5vdCwgdGhlIGJlaGF2aW9yIGlzIGRldGVybWluZWQgYnkgdGhlXG4gKiBgbWlzbWF0Y2hNb2RlYCBhcmd1bWVudC5cbiAqXG4gKiBUaGUgbmVzdGVkIHN0cnVjdHVyZSBvZiB0aGUgYGl0ZXJhdG9yc2AgYXJndW1lbnQgZGV0ZXJtaW5lcyB0aGVcbiAqIHN0cnVjdHVyZSBvZiBlbGVtZW50cyBpbiB0aGUgcmVzdWx0aW5nIGl0ZXJhdG9yLlxuICpcbiAqIEBwYXJhbSBpdGVyYXRvcnM6IEFuIGFycmF5IG9yIG9iamVjdCBjb250YWluaW5nIExhenlJdGVyYXRvcnMgYXQgdGhlXG4gKiBsZWF2ZXMuXG4gKiBAcGFyYW0gbWlzbWF0Y2hNb2RlOiBEZXRlcm1pbmVzIHdoYXQgdG8gZG8gd2hlbiBvbmUgdW5kZXJseWluZyBpdGVyYXRvclxuICogaXMgZXhoYXVzdGVkIGJlZm9yZSB0aGUgb3RoZXJzLiAgYFppcE1pc21hdGNoTW9kZS5GQUlMYCAodGhlIGRlZmF1bHQpXG4gKiBjYXVzZXMgYW4gZXJyb3IgdG8gYmUgdGhyb3duIGluIHRoaXMgY2FzZS4gIGBaaXBNaXNtYXRjaE1vZGUuU0hPUlRFU1RgXG4gKiBjYXVzZXMgdGhlIHppcHBlZCBpdGVyYXRvciB0byB0ZXJtaW5hdGUgd2l0aCB0aGUgZnVyc3QgdW5kZXJseWluZ1xuICogc3RyZWFtcywgc28gZWxlbWVudHMgcmVtYWluaW5nIG9uIHRoZSBsb25nZXIgc3RyZWFtcyBhcmUgaWdub3JlZC5cbiAqIGBaaXBNaXNtYXRjaE1vZGUuTE9OR0VTVGAgY2F1c2VzIHRoZSB6aXBwZWQgc3RyZWFtIHRvIGNvbnRpbnVlLCBmaWxsaW5nXG4gKiBpbiBudWxscyBmb3IgdGhlIGV4aGF1c3RlZCBzdHJlYW1zLCB1bnRpbCBhbGwgc3RyZWFtcyBhcmUgZXhoYXVzdGVkLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaXRlcmF0b3JGcm9tWmlwcGVkPE8gZXh0ZW5kcyB0Zi5UZW5zb3JDb250YWluZXI+KFxuICAgIGl0ZXJhdG9yczogSXRlcmF0b3JDb250YWluZXIsXG4gICAgbWlzbWF0Y2hNb2RlOiBaaXBNaXNtYXRjaE1vZGUgPSBaaXBNaXNtYXRjaE1vZGUuRkFJTCk6IExhenlJdGVyYXRvcjxPPiB7XG4gIHJldHVybiBuZXcgWmlwSXRlcmF0b3I8Tz4oaXRlcmF0b3JzLCBtaXNtYXRjaE1vZGUpO1xufVxuXG4vKipcbiAqIEFuIGFzeW5jaHJvbm91cyBpdGVyYXRvciwgcHJvdmlkaW5nIGxhenkgYWNjZXNzIHRvIGEgcG90ZW50aWFsbHlcbiAqIHVuYm91bmRlZCBzdHJlYW0gb2YgZWxlbWVudHMuXG4gKlxuICogSXRlcmF0b3IgY2FuIGJlIG9idGFpbmVkIGZyb20gYSBkYXRhc2V0OlxuICogYGNvbnN0IGl0ZXIgPSBhd2FpdCBkYXRhc2V0Lml0ZXJhdG9yKCk7YFxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgTGF6eUl0ZXJhdG9yPFQ+IHtcbiAgLy8gVGhpcyBjbGFzcyBpbXBsZW1lbnRzIEFzeW5jSXRlcmF0b3I8VD4sIGJ1dCB3ZSBoYXZlIG5vdCB5ZXQgc2V0IHRoZVxuICAvLyBUeXBlU2NyaXB0IC0tZG93bmxldmVsSXRlcmF0aW9uIGZsYWcgdG8gZW5hYmxlIHRoYXQuXG5cbiAgYWJzdHJhY3Qgc3VtbWFyeSgpOiBzdHJpbmc7XG5cbiAgLyoqXG4gICAqIFJldHVybnMgYSBgUHJvbWlzZWAgZm9yIHRoZSBuZXh0IGVsZW1lbnQgaW4gdGhlIHN0cmVhbS5cbiAgICpcbiAgICogV2hlbiBhbiBpdGVtIGNhbiBiZSBwcm92aWRlZCBzdWNjZXNzZnVsbHksIHRoZSByZXR1cm4gdmFsdWUgaXNcbiAgICogYHt2YWx1ZTpULCBkb25lOmZhbHNlfWAuXG4gICAqXG4gICAqIENhbGxpbmcgbmV4dCgpIG9uIGEgY2xvc2VkIHN0cmVhbSByZXR1cm5zIGB7dmFsdWU6bnVsbCwgZG9uZTp0cnVlfWAuXG4gICAqL1xuICBhYnN0cmFjdCBuZXh0KCk6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VD4+O1xuXG4gIC8qKlxuICAgKiBDb2xsZWN0IGFsbCByZW1haW5pbmcgZWxlbWVudHMgb2YgYSBib3VuZGVkIHN0cmVhbSBpbnRvIGFuIGFycmF5LlxuICAgKiBPYnZpb3VzbHkgdGhpcyB3aWxsIHN1Y2NlZWQgb25seSBmb3Igc21hbGwgc3RyZWFtcyB0aGF0IGZpdCBpbiBtZW1vcnkuXG4gICAqIFVzZWZ1bCBmb3IgdGVzdGluZy5cbiAgICpcbiAgICogQHJldHVybnMgQSBQcm9taXNlIGZvciBhbiBhcnJheSBvZiBzdHJlYW0gZWxlbWVudHMsIHdoaWNoIHdpbGwgcmVzb2x2ZVxuICAgKiAgIHdoZW4gdGhlIHN0cmVhbSBpcyBleGhhdXN0ZWQuXG4gICAqL1xuICBhc3luYyB0b0FycmF5KCk6IFByb21pc2U8VFtdPiB7XG4gICAgY29uc3QgcmVzdWx0OiBUW10gPSBbXTtcbiAgICBsZXQgeCA9IGF3YWl0IHRoaXMubmV4dCgpO1xuICAgIHdoaWxlICgheC5kb25lKSB7XG4gICAgICByZXN1bHQucHVzaCh4LnZhbHVlKTtcbiAgICAgIHggPSBhd2FpdCB0aGlzLm5leHQoKTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb2xsZWN0IGFsbCBlbGVtZW50cyBvZiB0aGlzIGRhdGFzZXQgaW50byBhbiBhcnJheSB3aXRoIHByZWZldGNoaW5nIDEwMFxuICAgKiBlbGVtZW50cy4gVGhpcyBpcyB1c2VmdWwgZm9yIHRlc3RpbmcsIGJlY2F1c2UgdGhlIHByZWZldGNoIGNoYW5nZXMgdGhlXG4gICAqIG9yZGVyIGluIHdoaWNoIHRoZSBQcm9taXNlcyBhcmUgcmVzb2x2ZWQgYWxvbmcgdGhlIHByb2Nlc3NpbmcgcGlwZWxpbmUuXG4gICAqIFRoaXMgbWF5IGhlbHAgZXhwb3NlIGJ1Z3Mgd2hlcmUgcmVzdWx0cyBhcmUgZGVwZW5kZW50IG9uIHRoZSBvcmRlciBvZlxuICAgKiBQcm9taXNlIHJlc29sdXRpb24gcmF0aGVyIHRoYW4gb24gdGhlIGxvZ2ljYWwgb3JkZXIgb2YgdGhlIHN0cmVhbSAoaS5lLixcbiAgICogZHVlIHRvIGhpZGRlbiBtdXRhYmxlIHN0YXRlKS5cbiAgICpcbiAgICogQHJldHVybnMgQSBQcm9taXNlIGZvciBhbiBhcnJheSBvZiBzdHJlYW0gZWxlbWVudHMsIHdoaWNoIHdpbGwgcmVzb2x2ZVxuICAgKiAgIHdoZW4gdGhlIHN0cmVhbSBpcyBleGhhdXN0ZWQuXG4gICAqL1xuICBhc3luYyB0b0FycmF5Rm9yVGVzdCgpOiBQcm9taXNlPFRbXT4ge1xuICAgIGNvbnN0IHN0cmVhbSA9IHRoaXMucHJlZmV0Y2goMTAwKTtcbiAgICBjb25zdCByZXN1bHQ6IFRbXSA9IFtdO1xuICAgIGxldCB4ID0gYXdhaXQgc3RyZWFtLm5leHQoKTtcbiAgICB3aGlsZSAoIXguZG9uZSkge1xuICAgICAgcmVzdWx0LnB1c2goeC52YWx1ZSk7XG4gICAgICB4ID0gYXdhaXQgc3RyZWFtLm5leHQoKTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIC8qKlxuICAgKiBEcmF3IGl0ZW1zIGZyb20gdGhlIHN0cmVhbSB1bnRpbCBpdCBpcyBleGhhdXN0ZWQuXG4gICAqXG4gICAqIFRoaXMgY2FuIGJlIHVzZWZ1bCB3aGVuIHRoZSBzdHJlYW0gaGFzIHNpZGUgZWZmZWN0cyBidXQgbm8gb3V0cHV0LiAgSW5cbiAgICogdGhhdCBjYXNlLCBjYWxsaW5nIHRoaXMgZnVuY3Rpb24gZ3VhcmFudGVlcyB0aGF0IHRoZSBzdHJlYW0gd2lsbCBiZVxuICAgKiBmdWxseSBwcm9jZXNzZWQuXG4gICAqL1xuICBhc3luYyByZXNvbHZlRnVsbHkoKTogUHJvbWlzZTx2b2lkPiB7XG4gICAgbGV0IHggPSBhd2FpdCB0aGlzLm5leHQoKTtcbiAgICB3aGlsZSAoIXguZG9uZSkge1xuICAgICAgeCA9IGF3YWl0IHRoaXMubmV4dCgpO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBEcmF3IGl0ZW1zIGZyb20gdGhlIHN0cmVhbSB1bnRpbCBpdCBpcyBleGhhdXN0ZWQsIG9yIGEgcHJlZGljYXRlIGZhaWxzLlxuICAgKlxuICAgKiBUaGlzIGNhbiBiZSB1c2VmdWwgd2hlbiB0aGUgc3RyZWFtIGhhcyBzaWRlIGVmZmVjdHMgYnV0IG5vIG91dHB1dC4gIEluXG4gICAqIHRoYXQgY2FzZSwgY2FsbGluZyB0aGlzIGZ1bmN0aW9uIGd1YXJhbnRlZXMgdGhhdCB0aGUgc3RyZWFtIHdpbGwgYmVcbiAgICogZnVsbHkgcHJvY2Vzc2VkLlxuICAgKi9cbiAgYXN5bmMgcmVzb2x2ZVdoaWxlKHByZWRpY2F0ZTogKHI6IFQpID0+IGJvb2xlYW4pOiBQcm9taXNlPHZvaWQ+IHtcbiAgICBsZXQgeCA9IGF3YWl0IHRoaXMubmV4dCgpO1xuICAgIGxldCBzaG91bGRDb250aW51ZSA9IHByZWRpY2F0ZSh4LnZhbHVlKTtcbiAgICB3aGlsZSAoKCF4LmRvbmUpICYmIHNob3VsZENvbnRpbnVlKSB7XG4gICAgICB4ID0gYXdhaXQgdGhpcy5uZXh0KCk7XG4gICAgICBzaG91bGRDb250aW51ZSA9IHByZWRpY2F0ZSh4LnZhbHVlKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogSGFuZGxlcyBlcnJvcnMgdGhyb3duIG9uIHRoaXMgc3RyZWFtIHVzaW5nIGEgcHJvdmlkZWQgaGFuZGxlciBmdW5jdGlvbi5cbiAgICpcbiAgICogQHBhcmFtIGhhbmRsZXIgQSBmdW5jdGlvbiB0aGF0IGhhbmRsZXMgYW55IGBFcnJvcmAgdGhyb3duIGR1cmluZyBhIGBuZXh0KClgXG4gICAqICAgY2FsbCBhbmQgcmV0dXJucyB0cnVlIGlmIHRoZSBzdHJlYW0gc2hvdWxkIGNvbnRpbnVlIChkcm9wcGluZyB0aGUgZmFpbGVkXG4gICAqICAgY2FsbCkgb3IgZmFsc2UgaWYgdGhlIHN0cmVhbSBzaG91bGQgcXVpZXRseSB0ZXJtaW5hdGUuICBJZiB0aGUgaGFuZGxlclxuICAgKiAgIGl0c2VsZiB0aHJvd3MgKG9yIHJldGhyb3dzKSBhbiBgRXJyb3JgLCB0aGF0IHdpbGwgYmUgcHJvcGFnYXRlZC5cbiAgICpcbiAgICogQHJldHVybnMgQSBgTGF6eUl0ZXJhdG9yYCBvZiBlbGVtZW50cyBwYXNzZWQgdGhyb3VnaCBmcm9tIHVwc3RyZWFtLFxuICAgKiAgIHBvc3NpYmx5IGZpbHRlcmluZyBvciB0ZXJtaW5hdGluZyBvbiB1cHN0cmVhbSBgbmV4dCgpYCBjYWxscyB0aGF0XG4gICAqICAgdGhyb3cgYW4gYEVycm9yYC5cbiAgICovXG4gIGhhbmRsZUVycm9ycyhoYW5kbGVyOiAoZXJyb3I6IEVycm9yKSA9PiBib29sZWFuKTogTGF6eUl0ZXJhdG9yPFQ+IHtcbiAgICByZXR1cm4gbmV3IEVycm9ySGFuZGxpbmdMYXp5SXRlcmF0b3IodGhpcywgaGFuZGxlcik7XG4gIH1cblxuICAvLyBUT0RPKHNvZXJnZWwpOiBJbXBsZW1lbnQgcmVkdWNlKCkgZXRjLlxuXG4gIC8qKlxuICAgKiBGaWx0ZXJzIHRoaXMgc3RyZWFtIGFjY29yZGluZyB0byBgcHJlZGljYXRlYC5cbiAgICpcbiAgICogQHBhcmFtIHByZWRpY2F0ZSBBIGZ1bmN0aW9uIG1hcHBpbmcgYSBzdHJlYW0gZWxlbWVudCB0byBhIGJvb2xlYW4gb3IgYVxuICAgKiBgUHJvbWlzZWAgZm9yIG9uZS5cbiAgICpcbiAgICogQHJldHVybnMgQSBgTGF6eUl0ZXJhdG9yYCBvZiBlbGVtZW50cyBmb3Igd2hpY2ggdGhlIHByZWRpY2F0ZSB3YXMgdHJ1ZS5cbiAgICovXG4gIGZpbHRlcihwcmVkaWNhdGU6ICh2YWx1ZTogVCkgPT4gYm9vbGVhbik6IExhenlJdGVyYXRvcjxUPiB7XG4gICAgcmV0dXJuIG5ldyBGaWx0ZXJJdGVyYXRvcih0aGlzLCBwcmVkaWNhdGUpO1xuICB9XG5cbiAgLyoqXG4gICAqIE1hcHMgdGhpcyBzdHJlYW0gdGhyb3VnaCBhIDEtdG8tMSB0cmFuc2Zvcm0uXG4gICAqXG4gICAqIEBwYXJhbSB0cmFuc2Zvcm0gQSBmdW5jdGlvbiBtYXBwaW5nIGEgc3RyZWFtIGVsZW1lbnQgdG8gYSB0cmFuc2Zvcm1lZFxuICAgKiAgIGVsZW1lbnQuXG4gICAqXG4gICAqIEByZXR1cm5zIEEgYExhenlJdGVyYXRvcmAgb2YgdHJhbnNmb3JtZWQgZWxlbWVudHMuXG4gICAqL1xuICBtYXA8Tz4odHJhbnNmb3JtOiAodmFsdWU6IFQpID0+IE8pOiBMYXp5SXRlcmF0b3I8Tz4ge1xuICAgIHJldHVybiBuZXcgTWFwSXRlcmF0b3IodGhpcywgdHJhbnNmb3JtKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBNYXBzIHRoaXMgc3RyZWFtIHRocm91Z2ggYW4gYXN5bmMgMS10by0xIHRyYW5zZm9ybS5cbiAgICpcbiAgICogQHBhcmFtIHRyYW5zZm9ybSBBIGZ1bmN0aW9uIG1hcHBpbmcgYSBzdHJlYW0gZWxlbWVudCB0byBhIGBQcm9taXNlYCBmb3IgYVxuICAgKiAgIHRyYW5zZm9ybWVkIHN0cmVhbSBlbGVtZW50LlxuICAgKlxuICAgKiBAcmV0dXJucyBBIGBMYXp5SXRlcmF0b3JgIG9mIHRyYW5zZm9ybWVkIGVsZW1lbnRzLlxuICAgKi9cbiAgbWFwQXN5bmM8Tz4odHJhbnNmb3JtOiAodmFsdWU6IFQpID0+IFByb21pc2U8Tz4pOiBMYXp5SXRlcmF0b3I8Tz4ge1xuICAgIHJldHVybiBuZXcgQXN5bmNNYXBJdGVyYXRvcih0aGlzLCB0cmFuc2Zvcm0pO1xuICB9XG5cbiAgLyoqXG4gICAqIE1hcHMgdGhpcyBzdHJlYW0gdGhyb3VnaCBhIDEtdG8tMSB0cmFuc2Zvcm0sIGZvcmNpbmcgc2VyaWFsIGV4ZWN1dGlvbi5cbiAgICpcbiAgICogQHBhcmFtIHRyYW5zZm9ybSBBIGZ1bmN0aW9uIG1hcHBpbmcgYSBzdHJlYW0gZWxlbWVudCB0byBhIHRyYW5zZm9ybWVkXG4gICAqICAgZWxlbWVudC5cbiAgICpcbiAgICogQHJldHVybnMgQSBgTGF6eUl0ZXJhdG9yYCBvZiB0cmFuc2Zvcm1lZCBlbGVtZW50cy5cbiAgICovXG4gIHNlcmlhbE1hcEFzeW5jPE8+KHRyYW5zZm9ybTogKHZhbHVlOiBUKSA9PiBQcm9taXNlPE8+KTogTGF6eUl0ZXJhdG9yPE8+IHtcbiAgICByZXR1cm4gbmV3IEFzeW5jTWFwSXRlcmF0b3IodGhpcywgdHJhbnNmb3JtKS5zZXJpYWwoKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBNYXBzIHRoaXMgc3RyZWFtIHRocm91Z2ggYSAxLXRvLW1hbnkgdHJhbnNmb3JtLlxuICAgKlxuICAgKiBAcGFyYW0gdHJhbnNmb3JtIEEgZnVuY3Rpb24gbWFwcGluZyBhIHN0cmVhbSBlbGVtZW50IHRvIGFuIGFycmF5IG9mXG4gICAqICAgdHJhbnNmb3JtZWQgZWxlbWVudHMuXG4gICAqXG4gICAqIEByZXR1cm5zIEEgYERhdGFTdHJlYW1gIG9mIHRyYW5zZm9ybWVkIGVsZW1lbnRzLlxuICAgKi9cbiAgZmxhdG1hcDxPPih0cmFuc2Zvcm06ICh2YWx1ZTogVCkgPT4gT1tdKTogTGF6eUl0ZXJhdG9yPE8+IHtcbiAgICByZXR1cm4gbmV3IEZsYXRtYXBJdGVyYXRvcih0aGlzLCB0cmFuc2Zvcm0pO1xuICB9XG5cbiAgLyoqXG4gICAqIEFwcGx5IGEgZnVuY3Rpb24gdG8gZXZlcnkgZWxlbWVudCBvZiB0aGUgc3RyZWFtLlxuICAgKlxuICAgKiBAcGFyYW0gZiBBIGZ1bmN0aW9uIHRvIGFwcGx5IHRvIGVhY2ggc3RyZWFtIGVsZW1lbnQuXG4gICAqL1xuICBhc3luYyBmb3JFYWNoQXN5bmMoZjogKHZhbHVlOiBUKSA9PiB2b2lkKTogUHJvbWlzZTx2b2lkPiB7XG4gICAgcmV0dXJuIHRoaXMubWFwKGYpLnJlc29sdmVGdWxseSgpO1xuICB9XG5cbiAgLyoqXG4gICAqIEFwcGx5IGEgZnVuY3Rpb24gdG8gZXZlcnkgZWxlbWVudCBvZiB0aGUgc3RyZWFtLCBmb3JjaW5nIHNlcmlhbCBleGVjdXRpb24uXG4gICAqXG4gICAqIEBwYXJhbSBmIEEgZnVuY3Rpb24gdG8gYXBwbHkgdG8gZWFjaCBzdHJlYW0gZWxlbWVudC4gIFNob3VsZCByZXR1cm4gJ3RydWUnXG4gICAqICAgdG8gaW5kaWNhdGUgdGhhdCB0aGUgc3RyZWFtIHNob3VsZCBjb250aW51ZSwgb3IgJ2ZhbHNlJyB0byBjYXVzZSBpdCB0b1xuICAgKiAgIHRlcm1pbmF0ZS5cbiAgICovXG4gIGFzeW5jIHNlcmlhbEZvckVhY2goZjogKHZhbHVlOiBUKSA9PiBQcm9taXNlPGJvb2xlYW4+KTogUHJvbWlzZTx2b2lkPiB7XG4gICAgcmV0dXJuIHRoaXMuc2VyaWFsTWFwQXN5bmMoZikucmVzb2x2ZVdoaWxlKHggPT4gKHggPT09IHRydWUpKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBHcm91cHMgZWxlbWVudHMgaW50byBiYXRjaGVzLCByZXByZXNlbnRlZCBhcyBhcnJheXMgb2YgZWxlbWVudHMuXG4gICAqXG4gICAqIFdlIGNhbiB0aGluayBvZiB0aGUgZWxlbWVudHMgb2YgdGhpcyBpdGVyYXRvciBhcyAncm93cycgKGV2ZW4gaWYgdGhleSBhcmVcbiAgICogbmVzdGVkIHN0cnVjdHVyZXMpLiAgQnkgdGhlIHNhbWUgdG9rZW4sIGNvbnNlY3V0aXZlIHZhbHVlcyBmb3IgYSBnaXZlblxuICAgKiBrZXkgd2l0aGluIHRoZSBlbGVtZW50cyBmb3JtIGEgJ2NvbHVtbicuICBUaGlzIG1hdGNoZXMgdGhlIHVzdWFsIHNlbnNlIG9mXG4gICAqICdyb3cnIGFuZCAnY29sdW1uJyB3aGVuIHByb2Nlc3NpbmcgdGFidWxhciBkYXRhIChlLmcuLCBwYXJzaW5nIGEgQ1NWKS5cbiAgICpcbiAgICogVGh1cywgXCJSb3ctbWFqb3JcIiBtZWFucyB0aGF0IHRoZSByZXN1bHRpbmcgYmF0Y2ggaXMgc2ltcGx5IGEgY29sbGVjdGlvbiBvZlxuICAgKiByb3dzOiBgW3JvdzEsIHJvdzIsIHJvdzMsIC4uLl1gLiAgVGhpcyBpcyBjb250cmFzdCB0byB0aGUgY29sdW1uLW1ham9yXG4gICAqIGZvcm0sIHdoaWNoIGlzIG5lZWRlZCBmb3IgdmVjdG9yaXplZCBjb21wdXRhdGlvbi5cbiAgICpcbiAgICogQHBhcmFtIGJhdGNoU2l6ZSBUaGUgbnVtYmVyIG9mIGVsZW1lbnRzIGRlc2lyZWQgcGVyIGJhdGNoLlxuICAgKiBAcGFyYW0gc21hbGxMYXN0QmF0Y2ggV2hldGhlciB0byBlbWl0IHRoZSBmaW5hbCBiYXRjaCB3aGVuIGl0IGhhcyBmZXdlclxuICAgKiAgIHRoYW4gYmF0Y2hTaXplIGVsZW1lbnRzLiBEZWZhdWx0IHRydWUuXG4gICAqIEByZXR1cm5zIEEgYExhenlJdGVyYXRvcmAgb2YgYmF0Y2hlcyBvZiBlbGVtZW50cywgcmVwcmVzZW50ZWQgYXMgYXJyYXlzXG4gICAqICAgb2YgdGhlIG9yaWdpbmFsIGVsZW1lbnQgdHlwZS5cbiAgICovXG4gIHJvd01ham9yQmF0Y2goYmF0Y2hTaXplOiBudW1iZXIsIHNtYWxsTGFzdEJhdGNoID0gdHJ1ZSk6IExhenlJdGVyYXRvcjxUW10+IHtcbiAgICByZXR1cm4gbmV3IFJvd01ham9yQmF0Y2hJdGVyYXRvcih0aGlzLCBiYXRjaFNpemUsIHNtYWxsTGFzdEJhdGNoKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBHcm91cHMgZWxlbWVudHMgaW50byBiYXRjaGVzLCByZXByZXNlbnRlZCBpbiBjb2x1bW4tbWFqb3IgZm9ybS5cbiAgICpcbiAgICogV2UgY2FuIHRoaW5rIG9mIHRoZSBlbGVtZW50cyBvZiB0aGlzIGl0ZXJhdG9yIGFzICdyb3dzJyAoZXZlbiBpZiB0aGV5IGFyZVxuICAgKiBuZXN0ZWQgc3RydWN0dXJlcykuICBCeSB0aGUgc2FtZSB0b2tlbiwgY29uc2VjdXRpdmUgdmFsdWVzIGZvciBhIGdpdmVuXG4gICAqIGtleSB3aXRoaW4gdGhlIGVsZW1lbnRzIGZvcm0gYSAnY29sdW1uJy4gIFRoaXMgbWF0Y2hlcyB0aGUgdXN1YWwgc2Vuc2Ugb2ZcbiAgICogJ3JvdycgYW5kICdjb2x1bW4nIHdoZW4gcHJvY2Vzc2luZyB0YWJ1bGFyIGRhdGEgKGUuZy4sIHBhcnNpbmcgYSBDU1YpLlxuICAgKlxuICAgKiBUaHVzLCBcImNvbHVtbi1tYWpvclwiIG1lYW5zIHRoYXQgdGhlIHJlc3VsdGluZyBiYXRjaCBpcyBhIChwb3RlbnRpYWxseVxuICAgKiBuZXN0ZWQpIHN0cnVjdHVyZSByZXByZXNlbnRpbmcgdGhlIGNvbHVtbnMuICBFYWNoIGNvbHVtbiBlbnRyeSwgdGhlbixcbiAgICogY29udGFpbnMgYSBjb2xsZWN0aW9uIG9mIHRoZSB2YWx1ZXMgZm91bmQgaW4gdGhhdCBjb2x1bW4gZm9yIGEgcmFuZ2Ugb2ZcbiAgICogaW5wdXQgZWxlbWVudHMuICBUaGlzIHJlcHJlc2VudGF0aW9uIGFsbG93cyBmb3IgdmVjdG9yaXplZCBjb21wdXRhdGlvbiwgaW5cbiAgICogY29udHJhc3QgdG8gdGhlIHJvdy1tYWpvciBmb3JtLlxuICAgKlxuICAgKiBUaGUgaW5wdXRzIHNob3VsZCBhbGwgaGF2ZSB0aGUgc2FtZSBuZXN0ZWQgc3RydWN0dXJlIChpLmUuLCBvZiBhcnJheXMgYW5kXG4gICAqIGRpY3RzKS4gIFRoZSByZXN1bHQgaXMgYSBzaW5nbGUgb2JqZWN0IHdpdGggdGhlIHNhbWUgbmVzdGVkIHN0cnVjdHVyZSxcbiAgICogd2hlcmUgdGhlIGxlYXZlcyBhcmUgYXJyYXlzIGNvbGxlY3RpbmcgdGhlIHZhbHVlcyBvZiB0aGUgaW5wdXRzIGF0IHRoYXRcbiAgICogbG9jYXRpb24gKG9yLCBvcHRpb25hbGx5LCB0aGUgcmVzdWx0IG9mIGEgY3VzdG9tIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhvc2VcbiAgICogYXJyYXlzKS5cbiAgICpcbiAgICogQHBhcmFtIGJhdGNoU2l6ZSBUaGUgbnVtYmVyIG9mIGVsZW1lbnRzIGRlc2lyZWQgcGVyIGJhdGNoLlxuICAgKiBAcGFyYW0gc21hbGxMYXN0QmF0Y2ggV2hldGhlciB0byBlbWl0IHRoZSBmaW5hbCBiYXRjaCB3aGVuIGl0IGhhcyBmZXdlclxuICAgKiAgIHRoYW4gYmF0Y2hTaXplIGVsZW1lbnRzLiBEZWZhdWx0IHRydWUuXG4gICAqIEBwYXJhbSB6aXBGbjogKG9wdGlvbmFsKSBBIGZ1bmN0aW9uIHRoYXQgZXhwZWN0cyBhbiBhcnJheSBvZiBlbGVtZW50cyBhdCBhXG4gICAqICAgc2luZ2xlIG5vZGUgb2YgdGhlIG9iamVjdCB0cmVlLCBhbmQgcmV0dXJucyBhIGBEZWVwTWFwUmVzdWx0YC4gIFRoZVxuICAgKiAgIGBEZWVwTWFwUmVzdWx0YCBlaXRoZXIgcHJvdmlkZXMgYSByZXN1bHQgdmFsdWUgZm9yIHRoYXQgbm9kZSAoaS5lLixcbiAgICogICByZXByZXNlbnRpbmcgdGhlIHN1YnRyZWUpLCBvciBpbmRpY2F0ZXMgdGhhdCB0aGUgbm9kZSBzaG91bGQgYmUgcHJvY2Vzc2VkXG4gICAqICAgcmVjdXJzaXZlbHkuICBUaGUgZGVmYXVsdCB6aXBGbiByZWN1cnNlcyBhcyBmYXIgYXMgcG9zc2libGUgYW5kIHBsYWNlc1xuICAgKiAgIGFycmF5cyBhdCB0aGUgbGVhdmVzLlxuICAgKiBAcmV0dXJucyBBIGBMYXp5SXRlcmF0b3JgIG9mIGJhdGNoZXMgb2YgZWxlbWVudHMsIHJlcHJlc2VudGVkIGFzIGFuIG9iamVjdFxuICAgKiAgIHdpdGggY29sbGVjdGlvbnMgYXQgdGhlIGxlYXZlcy5cbiAgICovXG4gIGNvbHVtbk1ham9yQmF0Y2goXG4gICAgICBiYXRjaFNpemU6IG51bWJlciwgc21hbGxMYXN0QmF0Y2ggPSB0cnVlLFxuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgemlwRm46ICh4czogYW55W10pID0+IERlZXBNYXBSZXN1bHQgPSB6aXBUb0xpc3QpOlxuICAgICAgTGF6eUl0ZXJhdG9yPHRmLlRlbnNvckNvbnRhaW5lcj4ge1xuICAgIC8vIEZpcnN0IGNvbGxlY3QgdGhlIGRlc2lyZWQgbnVtYmVyIG9mIGlucHV0IGVsZW1lbnRzIGFzIGEgcm93LW1ham9yIGJhdGNoLlxuICAgIGNvbnN0IHJvd0JhdGNoZXMgPSB0aGlzLnJvd01ham9yQmF0Y2goYmF0Y2hTaXplLCBzbWFsbExhc3RCYXRjaCk7XG4gICAgLy8gTm93ICdyb3RhdGUnIG9yICdwaXZvdCcgdGhlIGRhdGEsIGNvbGxlY3RpbmcgYWxsIHZhbHVlcyBmcm9tIGVhY2ggY29sdW1uXG4gICAgLy8gaW4gdGhlIGJhdGNoIChpLmUuLCBmb3IgZWFjaCBrZXkgd2l0aGluIHRoZSBlbGVtZW50cykgaW50byBhbiBhcnJheS5cbiAgICByZXR1cm4gcm93QmF0Y2hlcy5tYXAoeCA9PiBkZWVwWmlwKHgsIHppcEZuKSk7XG4gIH1cblxuICAvKipcbiAgICogQ29uY2F0ZW5hdGUgdGhpcyBgTGF6eUl0ZXJhdG9yYCB3aXRoIGFub3RoZXIuXG4gICAqXG4gICAqIEBwYXJhbSBpdGVyYXRvciBBIGBMYXp5SXRlcmF0b3JgIHRvIGJlIGNvbmNhdGVuYXRlZCBvbnRvIHRoaXMgb25lLlxuICAgKiBAcGFyYW0gYmFzZUVycm9ySGFuZGxlciBBbiBvcHRpb25hbCBmdW5jdGlvbiB0aGF0IGNhbiBpbnRlcmNlcHQgYEVycm9yYHNcbiAgICogICByYWlzZWQgZHVyaW5nIGEgYG5leHQoKWAgY2FsbCBvbiB0aGUgYmFzZSBzdHJlYW0uICBUaGlzIGZ1bmN0aW9uIGNhblxuICAgKiAgIGRlY2lkZSB3aGV0aGVyIHRoZSBlcnJvciBzaG91bGQgYmUgcHJvcGFnYXRlZCwgd2hldGhlciB0aGUgZXJyb3Igc2hvdWxkXG4gICAqICAgYmUgaWdub3JlZCwgb3Igd2hldGhlciB0aGUgYmFzZSBzdHJlYW0gc2hvdWxkIGJlIHRlcm1pbmF0ZWQuXG4gICAqIEByZXR1cm5zIEEgYExhenlJdGVyYXRvcmAuXG4gICAqL1xuICBjb25jYXRlbmF0ZShcbiAgICAgIGl0ZXJhdG9yOiBMYXp5SXRlcmF0b3I8VD4sXG4gICAgICBiYXNlRXJyb3JIYW5kbGVyPzogKGU6IEVycm9yKSA9PiBib29sZWFuKTogTGF6eUl0ZXJhdG9yPFQ+IHtcbiAgICByZXR1cm4gbmV3IENoYWluZWRJdGVyYXRvcihcbiAgICAgICAgaXRlcmF0b3JGcm9tSXRlbXMoW3RoaXMsIGl0ZXJhdG9yXSksIGJhc2VFcnJvckhhbmRsZXIpO1xuICB9XG5cbiAgLyoqXG4gICAqIExpbWl0cyB0aGlzIHN0cmVhbSB0byByZXR1cm4gYXQgbW9zdCBgY291bnRgIGl0ZW1zLlxuICAgKlxuICAgKiBAcGFyYW0gY291bnQgVGhlIG1heGltdW0gbnVtYmVyIG9mIGl0ZW1zIHRvIHByb3ZpZGUgZnJvbSB0aGUgc3RyZWFtLiBJZlxuICAgKiBhIG5lZ2F0aXZlIG9yIHVuZGVmaW5lZCB2YWx1ZSBpcyBnaXZlbiwgdGhlIGVudGlyZSBzdHJlYW0gaXMgcmV0dXJuZWRcbiAgICogICB1bmFsdGVyZWQuXG4gICAqL1xuICB0YWtlKGNvdW50OiBudW1iZXIpOiBMYXp5SXRlcmF0b3I8VD4ge1xuICAgIGlmIChjb3VudCA8IDAgfHwgY291bnQgPT0gbnVsbCkge1xuICAgICAgcmV0dXJuIHRoaXM7XG4gICAgfVxuICAgIHJldHVybiBuZXcgVGFrZUl0ZXJhdG9yKHRoaXMsIGNvdW50KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTa2lwcyB0aGUgZmlyc3QgYGNvdW50YCBpdGVtcyBpbiB0aGlzIHN0cmVhbS5cbiAgICpcbiAgICogQHBhcmFtIGNvdW50IFRoZSBudW1iZXIgb2YgaXRlbXMgdG8gc2tpcC4gIElmIGEgbmVnYXRpdmUgb3IgdW5kZWZpbmVkXG4gICAqIHZhbHVlIGlzIGdpdmVuLCB0aGUgZW50aXJlIHN0cmVhbSBpcyByZXR1cm5lZCB1bmFsdGVyZWQuXG4gICAqL1xuICBza2lwKGNvdW50OiBudW1iZXIpOiBMYXp5SXRlcmF0b3I8VD4ge1xuICAgIGlmIChjb3VudCA8IDAgfHwgY291bnQgPT0gbnVsbCkge1xuICAgICAgcmV0dXJuIHRoaXM7XG4gICAgfVxuICAgIHJldHVybiBuZXcgU2tpcEl0ZXJhdG9yKHRoaXMsIGNvdW50KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBQcmVmZXRjaCB0aGUgZmlyc3QgYGJ1ZmZlclNpemVgIGl0ZW1zIGluIHRoaXMgc3RyZWFtLlxuICAgKlxuICAgKiBOb3RlIHRoaXMgcHJlZmV0Y2hlcyBQcm9taXNlcywgYnV0IG1ha2VzIG5vIGd1YXJhbnRlZXMgYWJvdXQgd2hlbiB0aG9zZVxuICAgKiBQcm9taXNlcyByZXNvbHZlLlxuICAgKlxuICAgKiBAcGFyYW0gYnVmZmVyU2l6ZTogQW4gaW50ZWdlciBzcGVjaWZ5aW5nIHRoZSBudW1iZXIgb2YgZWxlbWVudHMgdG8gYmVcbiAgICogICBwcmVmZXRjaGVkLlxuICAgKi9cbiAgcHJlZmV0Y2goYnVmZmVyU2l6ZTogbnVtYmVyKTogTGF6eUl0ZXJhdG9yPFQ+IHtcbiAgICByZXR1cm4gbmV3IFByZWZldGNoSXRlcmF0b3IodGhpcywgYnVmZmVyU2l6ZSk7XG4gIH1cblxuICAvLyBUT0RPKHNvZXJnZWwpOiBkZWVwIHNoYXJkZWQgc2h1ZmZsZSwgd2hlcmUgc3VwcG9ydGVkXG5cbiAgLyoqXG4gICAqIFJhbmRvbWx5IHNodWZmbGVzIHRoZSBlbGVtZW50cyBvZiB0aGlzIHN0cmVhbS5cbiAgICpcbiAgICogQHBhcmFtIGJ1ZmZlclNpemU6IEFuIGludGVnZXIgc3BlY2lmeWluZyB0aGUgbnVtYmVyIG9mIGVsZW1lbnRzIGZyb21cbiAgICogdGhpcyBzdHJlYW0gZnJvbSB3aGljaCB0aGUgbmV3IHN0cmVhbSB3aWxsIHNhbXBsZS5cbiAgICogQHBhcmFtIHNlZWQ6IChPcHRpb25hbC4pIEFuIGludGVnZXIgc3BlY2lmeWluZyB0aGUgcmFuZG9tIHNlZWQgdGhhdFxuICAgKiB3aWxsIGJlIHVzZWQgdG8gY3JlYXRlIHRoZSBkaXN0cmlidXRpb24uXG4gICAqL1xuICBzaHVmZmxlKHdpbmRvd1NpemU6IG51bWJlciwgc2VlZD86IHN0cmluZyk6IExhenlJdGVyYXRvcjxUPiB7XG4gICAgcmV0dXJuIG5ldyBTaHVmZmxlSXRlcmF0b3IodGhpcywgd2luZG93U2l6ZSwgc2VlZCk7XG4gIH1cblxuICAvKipcbiAgICogRm9yY2UgYW4gaXRlcmF0b3IgdG8gZXhlY3V0ZSBzZXJpYWxseTogZWFjaCBuZXh0KCkgY2FsbCB3aWxsIGF3YWl0IHRoZVxuICAgKiBwcmlvciBvbmUsIHNvIHRoYXQgdGhleSBjYW5ub3QgZXhlY3V0ZSBjb25jdXJyZW50bHkuXG4gICAqL1xuICBzZXJpYWwoKTogTGF6eUl0ZXJhdG9yPFQ+IHtcbiAgICByZXR1cm4gbmV3IFNlcmlhbEl0ZXJhdG9yKHRoaXMpO1xuICB9XG59XG5cbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbi8vIFRoZSBmb2xsb3dpbmcgcHJpdmF0ZSBjbGFzc2VzIHNlcnZlIHRvIGltcGxlbWVudCB0aGUgY2hhaW5hYmxlIG1ldGhvZHNcbi8vIG9uIExhenlJdGVyYXRvci4gIFVuZm9ydHVuYXRlbHkgdGhleSBjYW4ndCBiZSBwbGFjZWQgaW4gc2VwYXJhdGUgZmlsZXMsXG4vLyBkdWUgdG8gcmVzdWx0aW5nIHRyb3VibGUgd2l0aCBjaXJjdWxhciBpbXBvcnRzLlxuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuXG4vLyBJdGVyYXRvcnMgdGhhdCBqdXN0IGV4dGVuZCBMYXp5SXRlcmF0b3IgZGlyZWN0bHlcbi8vID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cblxuY2xhc3MgQXJyYXlJdGVyYXRvcjxUPiBleHRlbmRzIExhenlJdGVyYXRvcjxUPiB7XG4gIHByaXZhdGUgdHJhdiA9IDA7XG4gIGNvbnN0cnVjdG9yKHByb3RlY3RlZCBpdGVtczogVFtdKSB7XG4gICAgc3VwZXIoKTtcbiAgfVxuXG4gIHN1bW1hcnkoKSB7XG4gICAgcmV0dXJuIGBBcnJheSBvZiAke3RoaXMuaXRlbXMubGVuZ3RofSBpdGVtc2A7XG4gIH1cblxuICBhc3luYyBuZXh0KCk6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VD4+IHtcbiAgICBpZiAodGhpcy50cmF2ID49IHRoaXMuaXRlbXMubGVuZ3RoKSB7XG4gICAgICByZXR1cm4ge3ZhbHVlOiBudWxsLCBkb25lOiB0cnVlfTtcbiAgICB9XG4gICAgY29uc3QgaXRlbSA9IHRoaXMuaXRlbXNbdGhpcy50cmF2XTtcbiAgICB0aGlzLnRyYXYrKztcbiAgICByZXR1cm4ge3ZhbHVlOiBkZWVwQ2xvbmUoaXRlbSksIGRvbmU6IGZhbHNlfTtcbiAgfVxufVxuXG5jbGFzcyBGdW5jdGlvbkNhbGxJdGVyYXRvcjxUPiBleHRlbmRzIExhenlJdGVyYXRvcjxUPiB7XG4gIGNvbnN0cnVjdG9yKFxuICAgICAgcHJvdGVjdGVkIG5leHRGbjogKCkgPT4gSXRlcmF0b3JSZXN1bHQ8VD58IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VD4+KSB7XG4gICAgc3VwZXIoKTtcbiAgfVxuXG4gIHN1bW1hcnkoKSB7XG4gICAgcmV0dXJuIGBGdW5jdGlvbiBjYWxsYDtcbiAgfVxuXG4gIGFzeW5jIG5leHQoKTogUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxUPj4ge1xuICAgIHRyeSB7XG4gICAgICByZXR1cm4gdGhpcy5uZXh0Rm4oKTtcbiAgICB9IGNhdGNoIChlKSB7XG4gICAgICAvLyBNb2RpZnkgdGhlIGVycm9yIG1lc3NhZ2UgYnV0IGxlYXZlIHRoZSBzdGFjayB0cmFjZSBpbnRhY3RcbiAgICAgIGUubWVzc2FnZSA9XG4gICAgICAgICAgYEVycm9yIHRocm93biB3aGlsZSBpdGVyYXRpbmcgdGhyb3VnaCBhIGRhdGFzZXQ6ICR7ZS5tZXNzYWdlfWA7XG4gICAgICB0aHJvdyBlO1xuICAgIH1cbiAgfVxufVxuXG5jbGFzcyBTZXJpYWxJdGVyYXRvcjxUPiBleHRlbmRzIExhenlJdGVyYXRvcjxUPiB7XG4gIC8vIFN0cmljdCBQcm9taXNlIGV4ZWN1dGlvbiBvcmRlcjpcbiAgLy8gYSBuZXh0KCkgY2FsbCBtYXkgbm90IGV2ZW4gYmVnaW4gdW50aWwgdGhlIHByZXZpb3VzIG9uZSBjb21wbGV0ZXMuXG4gIHByaXZhdGUgbGFzdFJlYWQ6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VD4+O1xuXG4gIGNvbnN0cnVjdG9yKHByb3RlY3RlZCB1cHN0cmVhbTogTGF6eUl0ZXJhdG9yPFQ+KSB7XG4gICAgc3VwZXIoKTtcbiAgICB0aGlzLmxhc3RSZWFkID0gUHJvbWlzZS5yZXNvbHZlKHt2YWx1ZTogbnVsbCwgZG9uZTogZmFsc2V9KTtcbiAgfVxuXG4gIHN1bW1hcnkoKSB7XG4gICAgcmV0dXJuIGAke3RoaXMudXBzdHJlYW0uc3VtbWFyeSgpfSAtPiBTZXJpYWxgO1xuICB9XG5cbiAgYXN5bmMgbmV4dCgpOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PiB7XG4gICAgLy8gVGhpcyBzZXRzIHRoaXMubGFzdFJlYWQgdG8gYSBuZXcgUHJvbWlzZSByaWdodCBhd2F5LCBhcyBvcHBvc2VkIHRvXG4gICAgLy8gc2F5aW5nIGBhd2FpdCB0aGlzLmxhc3RSZWFkOyB0aGlzLmxhc3RSZWFkID0gdGhpcy5zZXJpYWxOZXh0KCk7YCB3aGljaFxuICAgIC8vIHdvdWxkIG5vdCB3b3JrIGJlY2F1c2UgdGhpcy5uZXh0UmVhZCB3b3VsZCBiZSB1cGRhdGVkIG9ubHkgYWZ0ZXIgdGhlXG4gICAgLy8gcHJvbWlzZSByZXNvbHZlcy5cbiAgICB0aGlzLmxhc3RSZWFkID0gdGhpcy5sYXN0UmVhZC50aGVuKCgpID0+IHRoaXMuc2VyaWFsTmV4dCgpKTtcbiAgICByZXR1cm4gdGhpcy5sYXN0UmVhZDtcbiAgfVxuXG4gIHByaXZhdGUgYXN5bmMgc2VyaWFsTmV4dCgpOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PiB7XG4gICAgcmV0dXJuIHRoaXMudXBzdHJlYW0ubmV4dCgpO1xuICB9XG59XG5cbmNsYXNzIFNraXBJdGVyYXRvcjxUPiBleHRlbmRzIExhenlJdGVyYXRvcjxUPiB7XG4gIC8vIFN0cmljdCBQcm9taXNlIGV4ZWN1dGlvbiBvcmRlcjpcbiAgLy8gYSBuZXh0KCkgY2FsbCBtYXkgbm90IGV2ZW4gYmVnaW4gdW50aWwgdGhlIHByZXZpb3VzIG9uZSBjb21wbGV0ZXMuXG4gIHByaXZhdGUgbGFzdFJlYWQ6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VD4+O1xuXG4gIC8vIExvY2FsIHN0YXRlIHRoYXQgc2hvdWxkIG5vdCBiZSBjbG9iYmVyZWQgYnkgb3V0LW9mLW9yZGVyIGV4ZWN1dGlvbi5cbiAgY291bnQgPSAwO1xuXG4gIGNvbnN0cnVjdG9yKHByb3RlY3RlZCB1cHN0cmVhbTogTGF6eUl0ZXJhdG9yPFQ+LCBwcm90ZWN0ZWQgbWF4Q291bnQ6IG51bWJlcikge1xuICAgIHN1cGVyKCk7XG4gICAgdGhpcy5sYXN0UmVhZCA9IFByb21pc2UucmVzb2x2ZSh7dmFsdWU6IG51bGwsIGRvbmU6IGZhbHNlfSk7XG4gIH1cblxuICBzdW1tYXJ5KCkge1xuICAgIHJldHVybiBgJHt0aGlzLnVwc3RyZWFtLnN1bW1hcnkoKX0gLT4gU2tpcGA7XG4gIH1cblxuICBhc3luYyBuZXh0KCk6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VD4+IHtcbiAgICAvLyBUaGlzIHNldHMgdGhpcy5sYXN0UmVhZCB0byBhIG5ldyBQcm9taXNlIHJpZ2h0IGF3YXksIGFzIG9wcG9zZWQgdG9cbiAgICAvLyBzYXlpbmcgYGF3YWl0IHRoaXMubGFzdFJlYWQ7IHRoaXMubGFzdFJlYWQgPSB0aGlzLnNlcmlhbE5leHQoKTtgIHdoaWNoXG4gICAgLy8gd291bGQgbm90IHdvcmsgYmVjYXVzZSB0aGlzLm5leHRSZWFkIHdvdWxkIGJlIHVwZGF0ZWQgb25seSBhZnRlciB0aGVcbiAgICAvLyBwcm9taXNlIHJlc29sdmVzLlxuICAgIHRoaXMubGFzdFJlYWQgPSB0aGlzLmxhc3RSZWFkLnRoZW4oKCkgPT4gdGhpcy5zZXJpYWxOZXh0KCkpO1xuICAgIHJldHVybiB0aGlzLmxhc3RSZWFkO1xuICB9XG5cbiAgcHJpdmF0ZSBhc3luYyBzZXJpYWxOZXh0KCk6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VD4+IHtcbiAgICAvLyBUT0RPKHNvZXJnZWwpOiBjb25zaWRlciB0cmFkZW9mZnMgb2YgcmVhZGluZyBpbiBwYXJhbGxlbCwgZWcuXG4gICAgLy8gY29sbGVjdGluZyBuZXh0KCkgcHJvbWlzZXMgaW4gYW4gQXJyYXkgYW5kIHRoZW4gd2FpdGluZyBmb3JcbiAgICAvLyBQcm9taXNlLmFsbCgpIG9mIHRob3NlLiBCZW5lZml0OiBwc2V1ZG8tcGFyYWxsZWwgZXhlY3V0aW9uLiAgRHJhd2JhY2s6XG4gICAgLy8gbWF5YmUgZGVsYXllZCBHQy5cbiAgICB3aGlsZSAodGhpcy5jb3VudCsrIDwgdGhpcy5tYXhDb3VudCkge1xuICAgICAgY29uc3Qgc2tpcHBlZCA9IGF3YWl0IHRoaXMudXBzdHJlYW0ubmV4dCgpO1xuICAgICAgLy8gc2hvcnQtY2lyY3VpdCBpZiB1cHN0cmVhbSBpcyBhbHJlYWR5IGVtcHR5XG4gICAgICBpZiAoc2tpcHBlZC5kb25lKSB7XG4gICAgICAgIHJldHVybiBza2lwcGVkO1xuICAgICAgfVxuICAgICAgdGYuZGlzcG9zZShza2lwcGVkLnZhbHVlIGFzIHt9KTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMudXBzdHJlYW0ubmV4dCgpO1xuICB9XG59XG5cbmNsYXNzIFRha2VJdGVyYXRvcjxUPiBleHRlbmRzIExhenlJdGVyYXRvcjxUPiB7XG4gIGNvdW50ID0gMDtcbiAgY29uc3RydWN0b3IocHJvdGVjdGVkIHVwc3RyZWFtOiBMYXp5SXRlcmF0b3I8VD4sIHByb3RlY3RlZCBtYXhDb3VudDogbnVtYmVyKSB7XG4gICAgc3VwZXIoKTtcbiAgfVxuXG4gIHN1bW1hcnkoKSB7XG4gICAgcmV0dXJuIGAke3RoaXMudXBzdHJlYW0uc3VtbWFyeSgpfSAtPiBUYWtlYDtcbiAgfVxuXG4gIGFzeW5jIG5leHQoKTogUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxUPj4ge1xuICAgIGlmICh0aGlzLmNvdW50KysgPj0gdGhpcy5tYXhDb3VudCkge1xuICAgICAgcmV0dXJuIHt2YWx1ZTogbnVsbCwgZG9uZTogdHJ1ZX07XG4gICAgfVxuICAgIHJldHVybiB0aGlzLnVwc3RyZWFtLm5leHQoKTtcbiAgfVxufVxuXG4vLyBOb3RlIHRoaXMgYmF0Y2gganVzdCBncm91cHMgaXRlbXMgaW50byByb3ctd2lzZSBlbGVtZW50IGFycmF5cy5cbi8vIFJvdGF0aW5nIHRoZXNlIHRvIGEgY29sdW1uLXdpc2UgcmVwcmVzZW50YXRpb24gaGFwcGVucyBvbmx5IGF0IHRoZSBkYXRhc2V0XG4vLyBsZXZlbC5cbmNsYXNzIFJvd01ham9yQmF0Y2hJdGVyYXRvcjxUPiBleHRlbmRzIExhenlJdGVyYXRvcjxUW10+IHtcbiAgLy8gU3RyaWN0IFByb21pc2UgZXhlY3V0aW9uIG9yZGVyOlxuICAvLyBhIG5leHQoKSBjYWxsIG1heSBub3QgZXZlbiBiZWdpbiB1bnRpbCB0aGUgcHJldmlvdXMgb25lIGNvbXBsZXRlcy5cbiAgcHJpdmF0ZSBsYXN0UmVhZDogUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxUW10+PjtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIHByb3RlY3RlZCB1cHN0cmVhbTogTGF6eUl0ZXJhdG9yPFQ+LCBwcm90ZWN0ZWQgYmF0Y2hTaXplOiBudW1iZXIsXG4gICAgICBwcm90ZWN0ZWQgZW5hYmxlU21hbGxMYXN0QmF0Y2ggPSB0cnVlKSB7XG4gICAgc3VwZXIoKTtcbiAgICB0aGlzLmxhc3RSZWFkID0gUHJvbWlzZS5yZXNvbHZlKHt2YWx1ZTogbnVsbCwgZG9uZTogZmFsc2V9KTtcbiAgfVxuXG4gIHN1bW1hcnkoKSB7XG4gICAgcmV0dXJuIGAke3RoaXMudXBzdHJlYW0uc3VtbWFyeSgpfSAtPiBSb3dNYWpvckJhdGNoYDtcbiAgfVxuXG4gIGFzeW5jIG5leHQoKTogUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxUW10+PiB7XG4gICAgLy8gVGhpcyBzZXRzIHRoaXMubGFzdFJlYWQgdG8gYSBuZXcgUHJvbWlzZSByaWdodCBhd2F5LCBhcyBvcHBvc2VkIHRvXG4gICAgLy8gc2F5aW5nIGBhd2FpdCB0aGlzLmxhc3RSZWFkOyB0aGlzLmxhc3RSZWFkID0gdGhpcy5zZXJpYWxOZXh0KCk7YCB3aGljaFxuICAgIC8vIHdvdWxkIG5vdCB3b3JrIGJlY2F1c2UgdGhpcy5uZXh0UmVhZCB3b3VsZCBiZSB1cGRhdGVkIG9ubHkgYWZ0ZXIgdGhlXG4gICAgLy8gcHJvbWlzZSByZXNvbHZlcy5cbiAgICB0aGlzLmxhc3RSZWFkID0gdGhpcy5sYXN0UmVhZC50aGVuKCgpID0+IHRoaXMuc2VyaWFsTmV4dCgpKTtcbiAgICByZXR1cm4gdGhpcy5sYXN0UmVhZDtcbiAgfVxuXG4gIHByaXZhdGUgYXN5bmMgc2VyaWFsTmV4dCgpOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFRbXT4+IHtcbiAgICBjb25zdCBiYXRjaDogVFtdID0gW107XG4gICAgd2hpbGUgKGJhdGNoLmxlbmd0aCA8IHRoaXMuYmF0Y2hTaXplKSB7XG4gICAgICBjb25zdCBpdGVtID0gYXdhaXQgdGhpcy51cHN0cmVhbS5uZXh0KCk7XG4gICAgICBpZiAoaXRlbS5kb25lKSB7XG4gICAgICAgIGlmICh0aGlzLmVuYWJsZVNtYWxsTGFzdEJhdGNoICYmIGJhdGNoLmxlbmd0aCA+IDApIHtcbiAgICAgICAgICByZXR1cm4ge3ZhbHVlOiBiYXRjaCwgZG9uZTogZmFsc2V9O1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiB7dmFsdWU6IG51bGwsIGRvbmU6IHRydWV9O1xuICAgICAgfVxuICAgICAgYmF0Y2gucHVzaChpdGVtLnZhbHVlKTtcbiAgICB9XG4gICAgcmV0dXJuIHt2YWx1ZTogYmF0Y2gsIGRvbmU6IGZhbHNlfTtcbiAgfVxufVxuXG5jbGFzcyBGaWx0ZXJJdGVyYXRvcjxUPiBleHRlbmRzIExhenlJdGVyYXRvcjxUPiB7XG4gIC8vIFN0cmljdCBQcm9taXNlIGV4ZWN1dGlvbiBvcmRlcjpcbiAgLy8gYSBuZXh0KCkgY2FsbCBtYXkgbm90IGV2ZW4gYmVnaW4gdW50aWwgdGhlIHByZXZpb3VzIG9uZSBjb21wbGV0ZXMuXG4gIHByaXZhdGUgbGFzdFJlYWQ6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VD4+O1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgcHJvdGVjdGVkIHVwc3RyZWFtOiBMYXp5SXRlcmF0b3I8VD4sXG4gICAgICBwcm90ZWN0ZWQgcHJlZGljYXRlOiAodmFsdWU6IFQpID0+IGJvb2xlYW4pIHtcbiAgICBzdXBlcigpO1xuICAgIHRoaXMubGFzdFJlYWQgPSBQcm9taXNlLnJlc29sdmUoe3ZhbHVlOiBudWxsLCBkb25lOiBmYWxzZX0pO1xuICB9XG5cbiAgc3VtbWFyeSgpIHtcbiAgICByZXR1cm4gYCR7dGhpcy51cHN0cmVhbS5zdW1tYXJ5KCl9IC0+IEZpbHRlcmA7XG4gIH1cblxuICBhc3luYyBuZXh0KCk6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VD4+IHtcbiAgICAvLyBUaGlzIHNldHMgdGhpcy5sYXN0UmVhZCB0byBhIG5ldyBQcm9taXNlIHJpZ2h0IGF3YXksIGFzIG9wcG9zZWQgdG9cbiAgICAvLyBzYXlpbmcgYGF3YWl0IHRoaXMubGFzdFJlYWQ7IHRoaXMubGFzdFJlYWQgPSB0aGlzLnNlcmlhbE5leHQoKTtgIHdoaWNoXG4gICAgLy8gd291bGQgbm90IHdvcmsgYmVjYXVzZSB0aGlzLm5leHRSZWFkIHdvdWxkIGJlIHVwZGF0ZWQgb25seSBhZnRlciB0aGVcbiAgICAvLyBwcm9taXNlIHJlc29sdmVzLlxuICAgIHRoaXMubGFzdFJlYWQgPSB0aGlzLmxhc3RSZWFkLnRoZW4oKCkgPT4gdGhpcy5zZXJpYWxOZXh0KCkpO1xuICAgIHJldHVybiB0aGlzLmxhc3RSZWFkO1xuICB9XG5cbiAgcHJpdmF0ZSBhc3luYyBzZXJpYWxOZXh0KCk6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VD4+IHtcbiAgICB3aGlsZSAodHJ1ZSkge1xuICAgICAgY29uc3QgaXRlbSA9IGF3YWl0IHRoaXMudXBzdHJlYW0ubmV4dCgpO1xuICAgICAgaWYgKGl0ZW0uZG9uZSB8fCB0aGlzLnByZWRpY2F0ZShpdGVtLnZhbHVlKSkge1xuICAgICAgICByZXR1cm4gaXRlbTtcbiAgICAgIH1cbiAgICAgIHRmLmRpc3Bvc2UoaXRlbS52YWx1ZSBhcyB7fSk7XG4gICAgfVxuICB9XG59XG5cbmNsYXNzIE1hcEl0ZXJhdG9yPEksIE8+IGV4dGVuZHMgTGF6eUl0ZXJhdG9yPE8+IHtcbiAgY29uc3RydWN0b3IoXG4gICAgICBwcm90ZWN0ZWQgdXBzdHJlYW06IExhenlJdGVyYXRvcjxJPixcbiAgICAgIHByb3RlY3RlZCB0cmFuc2Zvcm06ICh2YWx1ZTogSSkgPT4gTykge1xuICAgIHN1cGVyKCk7XG4gIH1cblxuICBzdW1tYXJ5KCkge1xuICAgIHJldHVybiBgJHt0aGlzLnVwc3RyZWFtLnN1bW1hcnkoKX0gLT4gTWFwYDtcbiAgfVxuXG4gIGFzeW5jIG5leHQoKTogUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxPPj4ge1xuICAgIGNvbnN0IGl0ZW0gPSBhd2FpdCB0aGlzLnVwc3RyZWFtLm5leHQoKTtcbiAgICBpZiAoaXRlbS5kb25lKSB7XG4gICAgICByZXR1cm4ge3ZhbHVlOiBudWxsLCBkb25lOiB0cnVlfTtcbiAgICB9XG4gICAgY29uc3QgaW5wdXRUZW5zb3JzID0gdGYudGVuc29yX3V0aWwuZ2V0VGVuc29yc0luQ29udGFpbmVyKGl0ZW0udmFsdWUgYXMge30pO1xuICAgIC8vIENhcmVmdWw6IHRoZSB0cmFuc2Zvcm0gbWF5IG11dGF0ZSB0aGUgaXRlbSBpbiBwbGFjZS5cbiAgICAvLyBUaGF0J3Mgd2h5IHdlIGhhdmUgdG8gcmVtZW1iZXIgdGhlIGlucHV0IFRlbnNvcnMgYWJvdmUsIGFuZCB0aGVuXG4gICAgLy8gYmVsb3cgZGlzcG9zZSBvbmx5IHRob3NlIHRoYXQgd2VyZSBub3QgcGFzc2VkIHRocm91Z2ggdG8gdGhlIG91dHB1dC5cbiAgICAvLyBOb3RlIHRvbyB0aGF0IHRoZSB0cmFuc2Zvcm0gZnVuY3Rpb24gaXMgcmVzcG9uc2libGUgZm9yIHRpZHlpbmdcbiAgICAvLyBhbnkgaW50ZXJtZWRpYXRlIFRlbnNvcnMuICBIZXJlIHdlIGFyZSBjb25jZXJuZWQgb25seSBhYm91dCB0aGVcbiAgICAvLyBpbnB1dHMuXG4gICAgY29uc3QgbWFwcGVkID0gdGhpcy50cmFuc2Zvcm0oaXRlbS52YWx1ZSk7XG4gICAgY29uc3Qgb3V0cHV0VGVuc29ycyA9IHRmLnRlbnNvcl91dGlsLmdldFRlbnNvcnNJbkNvbnRhaW5lcihtYXBwZWQgYXMge30pO1xuXG4gICAgLy8gVE9ETyhzb2VyZ2VsKSBmYXN0ZXIgaW50ZXJzZWN0aW9uXG4gICAgLy8gVE9ETyhzb2VyZ2VsKSBtb3ZlIHRvIHRmLmRpc3Bvc2VFeGNlcHQoaW4sIG91dCk/XG4gICAgZm9yIChjb25zdCB0IG9mIGlucHV0VGVuc29ycykge1xuICAgICAgaWYgKCF0Zi50ZW5zb3JfdXRpbC5pc1RlbnNvckluTGlzdCh0LCBvdXRwdXRUZW5zb3JzKSkge1xuICAgICAgICB0LmRpc3Bvc2UoKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHt2YWx1ZTogbWFwcGVkLCBkb25lOiBmYWxzZX07XG4gIH1cbn1cblxuY2xhc3MgRXJyb3JIYW5kbGluZ0xhenlJdGVyYXRvcjxUPiBleHRlbmRzIExhenlJdGVyYXRvcjxUPiB7XG4gIGNvdW50ID0gMDtcbiAgY29uc3RydWN0b3IoXG4gICAgICBwcm90ZWN0ZWQgdXBzdHJlYW06IExhenlJdGVyYXRvcjxUPixcbiAgICAgIHByb3RlY3RlZCBoYW5kbGVyOiAoZXJyb3I6IEVycm9yKSA9PiBib29sZWFuKSB7XG4gICAgc3VwZXIoKTtcbiAgICB0aGlzLmxhc3RSZWFkID0gUHJvbWlzZS5yZXNvbHZlKHt2YWx1ZTogbnVsbCwgZG9uZTogZmFsc2V9KTtcbiAgfVxuXG4gIHN1bW1hcnkoKSB7XG4gICAgcmV0dXJuIGAke3RoaXMudXBzdHJlYW0uc3VtbWFyeSgpfSAtPiBoYW5kbGVFcnJvcnNgO1xuICB9XG5cbiAgLy8gU3RyaWN0IFByb21pc2UgZXhlY3V0aW9uIG9yZGVyOlxuICAvLyBhIG5leHQoKSBjYWxsIG1heSBub3QgZXZlbiBiZWdpbiB1bnRpbCB0aGUgcHJldmlvdXMgb25lIGNvbXBsZXRlcy5cbiAgcHJpdmF0ZSBsYXN0UmVhZDogUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxUPj47XG5cbiAgYXN5bmMgbmV4dCgpOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PiB7XG4gICAgLy8gVGhpcyBzZXRzIHRoaXMubGFzdFJlYWQgdG8gYSBuZXcgUHJvbWlzZSByaWdodCBhd2F5LCBhcyBvcHBvc2VkIHRvXG4gICAgLy8gc2F5aW5nIGBhd2FpdCB0aGlzLmxhc3RSZWFkOyB0aGlzLmxhc3RSZWFkID0gdGhpcy5zZXJpYWxOZXh0KCk7YCB3aGljaFxuICAgIC8vIHdvdWxkIG5vdCB3b3JrIGJlY2F1c2UgdGhpcy5uZXh0UmVhZCB3b3VsZCBiZSB1cGRhdGVkIG9ubHkgYWZ0ZXIgdGhlXG4gICAgLy8gcHJvbWlzZSByZXNvbHZlcy5cbiAgICB0aGlzLmxhc3RSZWFkID0gdGhpcy5sYXN0UmVhZC50aGVuKCgpID0+IHRoaXMuc2VyaWFsTmV4dCgpKTtcbiAgICByZXR1cm4gdGhpcy5sYXN0UmVhZDtcbiAgfVxuXG4gIGFzeW5jIHNlcmlhbE5leHQoKTogUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxUPj4ge1xuICAgIHdoaWxlICh0cnVlKSB7XG4gICAgICB0cnkge1xuICAgICAgICByZXR1cm4gYXdhaXQgdGhpcy51cHN0cmVhbS5uZXh0KCk7XG4gICAgICB9IGNhdGNoIChlKSB7XG4gICAgICAgIGlmICghdGhpcy5oYW5kbGVyKGUpKSB7XG4gICAgICAgICAgcmV0dXJuIHt2YWx1ZTogbnVsbCwgZG9uZTogdHJ1ZX07XG4gICAgICAgIH1cbiAgICAgICAgLy8gSWYgdGhlIGhhbmRsZXIgcmV0dXJucyB0cnVlLCBsb29wIGFuZCBmZXRjaCB0aGUgbmV4dCB1cHN0cmVhbSBpdGVtLlxuXG4gICAgICAgIC8vIElmIHRoZSB1cHN0cmVhbSBpdGVyYXRvciB0aHJvd3MgYW4gZW5kbGVzcyBzdHJlYW0gb2YgZXJyb3JzLCBhbmQgaWZcbiAgICAgICAgLy8gdGhlIGhhbmRsZXIgc2F5cyB0byBpZ25vcmUgdGhlbSwgdGhlbiB3ZSBsb29wIGZvcmV2ZXIgaGVyZS4gIFRoYXQgaXNcbiAgICAgICAgLy8gdGhlIGNvcnJlY3QgYmVoYXZpb3ItLSBpdCdzIHVwIHRvIHRoZSBoYW5kbGVyIHRvIGRlY2lkZSB3aGVuIHRvIHN0b3AuXG4gICAgICB9XG4gICAgfVxuICB9XG59XG5cbmNsYXNzIEFzeW5jTWFwSXRlcmF0b3I8SSwgTz4gZXh0ZW5kcyBMYXp5SXRlcmF0b3I8Tz4ge1xuICBjb25zdHJ1Y3RvcihcbiAgICAgIHByb3RlY3RlZCB1cHN0cmVhbTogTGF6eUl0ZXJhdG9yPEk+LFxuICAgICAgcHJvdGVjdGVkIHRyYW5zZm9ybTogKHZhbHVlOiBJKSA9PiBQcm9taXNlPE8+KSB7XG4gICAgc3VwZXIoKTtcbiAgfVxuXG4gIHN1bW1hcnkoKSB7XG4gICAgcmV0dXJuIGAke3RoaXMudXBzdHJlYW0uc3VtbWFyeSgpfSAtPiBBc3luY01hcGA7XG4gIH1cblxuICBhc3luYyBuZXh0KCk6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8Tz4+IHtcbiAgICBjb25zdCBpdGVtID0gYXdhaXQgdGhpcy51cHN0cmVhbS5uZXh0KCk7XG4gICAgaWYgKGl0ZW0uZG9uZSkge1xuICAgICAgcmV0dXJuIHt2YWx1ZTogbnVsbCwgZG9uZTogdHJ1ZX07XG4gICAgfVxuICAgIGNvbnN0IGlucHV0VGVuc29ycyA9IHRmLnRlbnNvcl91dGlsLmdldFRlbnNvcnNJbkNvbnRhaW5lcihpdGVtLnZhbHVlIGFzIHt9KTtcbiAgICAvLyBDYXJlZnVsOiB0aGUgdHJhbnNmb3JtIG1heSBtdXRhdGUgdGhlIGl0ZW0gaW4gcGxhY2UuXG4gICAgLy8gVGhhdCdzIHdoeSB3ZSBoYXZlIHRvIHJlbWVtYmVyIHRoZSBpbnB1dCBUZW5zb3JzIGFib3ZlLCBhbmQgdGhlblxuICAgIC8vIGJlbG93IGRpc3Bvc2Ugb25seSB0aG9zZSB0aGF0IHdlcmUgbm90IHBhc3NlZCB0aHJvdWdoIHRvIHRoZSBvdXRwdXQuXG4gICAgLy8gTm90ZSB0b28gdGhhdCB0aGUgdHJhbnNmb3JtIGZ1bmN0aW9uIGlzIHJlc3BvbnNpYmxlIGZvciB0aWR5aW5nXG4gICAgLy8gYW55IGludGVybWVkaWF0ZSBUZW5zb3JzLiAgSGVyZSB3ZSBhcmUgY29uY2VybmVkIG9ubHkgYWJvdXQgdGhlXG4gICAgLy8gaW5wdXRzLlxuICAgIGNvbnN0IG1hcHBlZCA9IGF3YWl0IHRoaXMudHJhbnNmb3JtKGl0ZW0udmFsdWUpO1xuICAgIGNvbnN0IG91dHB1dFRlbnNvcnMgPSB0Zi50ZW5zb3JfdXRpbC5nZXRUZW5zb3JzSW5Db250YWluZXIobWFwcGVkIGFzIHt9KTtcblxuICAgIC8vIFRPRE8oc29lcmdlbCkgZmFzdGVyIGludGVyc2VjdGlvblxuICAgIC8vIFRPRE8oc29lcmdlbCkgbW92ZSB0byB0Zi5kaXNwb3NlRXhjZXB0KGluLCBvdXQpP1xuICAgIGZvciAoY29uc3QgdCBvZiBpbnB1dFRlbnNvcnMpIHtcbiAgICAgIGlmICghdGYudGVuc29yX3V0aWwuaXNUZW5zb3JJbkxpc3QodCwgb3V0cHV0VGVuc29ycykpIHtcbiAgICAgICAgdC5kaXNwb3NlKCk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiB7dmFsdWU6IG1hcHBlZCwgZG9uZTogZmFsc2V9O1xuICB9XG59XG5cbi8vIEl0ZXJhdG9ycyB0aGF0IG1haW50YWluIGEgcXVldWUgb2YgcGVuZGluZyBpdGVtc1xuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuXG4vKipcbiAqIEEgYmFzZSBjbGFzcyBmb3IgdHJhbnNmb3JtaW5nIHN0cmVhbXMgdGhhdCBvcGVyYXRlIGJ5IG1haW50YWluaW5nIGFuXG4gKiBvdXRwdXQgcXVldWUgb2YgZWxlbWVudHMgdGhhdCBhcmUgcmVhZHkgdG8gcmV0dXJuIHZpYSBuZXh0KCkuICBUaGlzIGlzXG4gKiBjb21tb25seSByZXF1aXJlZCB3aGVuIHRoZSB0cmFuc2Zvcm1hdGlvbiBpcyAxLXRvLW1hbnk6ICBBIGNhbGwgdG8gbmV4dCgpXG4gKiBtYXkgdHJpZ2dlciBhIGNhbGwgdG8gdGhlIHVuZGVybHlpbmcgc3RyZWFtLCB3aGljaCB3aWxsIHByb2R1Y2UgbWFueVxuICogbWFwcGVkIGVsZW1lbnRzIG9mIHRoaXMgc3RyZWFtLS0gb2Ygd2hpY2ggd2UgbmVlZCB0byByZXR1cm4gb25seSBvbmUsIHNvXG4gKiB3ZSBoYXZlIHRvIHF1ZXVlIHRoZSByZXN0LlxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgT25lVG9NYW55SXRlcmF0b3I8VD4gZXh0ZW5kcyBMYXp5SXRlcmF0b3I8VD4ge1xuICAvLyBTdHJpY3QgUHJvbWlzZSBleGVjdXRpb24gb3JkZXI6XG4gIC8vIGEgbmV4dCgpIGNhbGwgbWF5IG5vdCBldmVuIGJlZ2luIHVudGlsIHRoZSBwcmV2aW91cyBvbmUgY29tcGxldGVzLlxuICBwcml2YXRlIGxhc3RSZWFkOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PjtcblxuICAvLyBMb2NhbCBzdGF0ZSB0aGF0IHNob3VsZCBub3QgYmUgY2xvYmJlcmVkIGJ5IG91dC1vZi1vcmRlciBleGVjdXRpb24uXG4gIHByb3RlY3RlZCBvdXRwdXRRdWV1ZTogUmluZ0J1ZmZlcjxUPjtcblxuICBjb25zdHJ1Y3RvcigpIHtcbiAgICBzdXBlcigpO1xuICAgIHRoaXMub3V0cHV0UXVldWUgPSBuZXcgR3Jvd2luZ1JpbmdCdWZmZXI8VD4oKTtcbiAgICB0aGlzLmxhc3RSZWFkID0gUHJvbWlzZS5yZXNvbHZlKHt2YWx1ZTogbnVsbCwgZG9uZTogZmFsc2V9KTtcbiAgfVxuXG4gIGFzeW5jIG5leHQoKTogUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxUPj4ge1xuICAgIC8vIFRoaXMgc2V0cyB0aGlzLmxhc3RSZWFkIHRvIGEgbmV3IFByb21pc2UgcmlnaHQgYXdheSwgYXMgb3Bwb3NlZCB0b1xuICAgIC8vIHNheWluZyBgYXdhaXQgdGhpcy5sYXN0UmVhZDsgdGhpcy5sYXN0UmVhZCA9IHRoaXMuc2VyaWFsTmV4dCgpO2Agd2hpY2hcbiAgICAvLyB3b3VsZCBub3Qgd29yayBiZWNhdXNlIHRoaXMubmV4dFJlYWQgd291bGQgYmUgdXBkYXRlZCBvbmx5IGFmdGVyIHRoZVxuICAgIC8vIHByb21pc2UgcmVzb2x2ZXMuXG4gICAgdGhpcy5sYXN0UmVhZCA9IHRoaXMubGFzdFJlYWQudGhlbigoKSA9PiB0aGlzLnNlcmlhbE5leHQoKSk7XG4gICAgcmV0dXJuIHRoaXMubGFzdFJlYWQ7XG4gIH1cblxuICAvKipcbiAgICogUmVhZCBvbmUgb3IgbW9yZSBjaHVua3MgZnJvbSB1cHN0cmVhbSBhbmQgcHJvY2VzcyB0aGVtLCBwb3NzaWJseVxuICAgKiByZWFkaW5nIG9yIHdyaXRpbmcgYSBjYXJyeW92ZXIsIGFuZCBhZGRpbmcgcHJvY2Vzc2VkIGl0ZW1zIHRvIHRoZVxuICAgKiBvdXRwdXQgcXVldWUuICBOb3RlIGl0J3MgcG9zc2libGUgdGhhdCBubyBpdGVtcyBhcmUgYWRkZWQgdG8gdGhlIHF1ZXVlXG4gICAqIG9uIGEgZ2l2ZW4gcHVtcCgpIGNhbGwsIGV2ZW4gaWYgdGhlIHVwc3RyZWFtIHN0cmVhbSBpcyBub3QgY2xvc2VkXG4gICAqIChlLmcuLCBiZWNhdXNlIGl0ZW1zIGFyZSBmaWx0ZXJlZCkuXG4gICAqXG4gICAqIEByZXR1cm4gYHRydWVgIGlmIGFueSBhY3Rpb24gd2FzIHRha2VuLCBpLmUuIGZldGNoaW5nIGl0ZW1zIGZyb20gdGhlXG4gICAqICAgdXBzdHJlYW0gc291cmNlIE9SIGFkZGluZyBpdGVtcyB0byB0aGUgb3V0cHV0IHF1ZXVlLiAgYGZhbHNlYCBpZiB0aGVcbiAgICogICB1cHN0cmVhbSBzb3VyY2UgaXMgZXhoYXVzdGVkIEFORCBub3RoaW5nIHdhcyBhZGRlZCB0byB0aGUgcXVldWVcbiAgICogKGkuZS4sIGFueSByZW1haW5pbmcgY2FycnlvdmVyKS5cbiAgICovXG4gIHByb3RlY3RlZCBhYnN0cmFjdCBwdW1wKCk6IFByb21pc2U8Ym9vbGVhbj47XG5cbiAgYXN5bmMgc2VyaWFsTmV4dCgpOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PiB7XG4gICAgLy8gRmV0Y2ggc28gdGhhdCB0aGUgcXVldWUgY29udGFpbnMgYXQgbGVhc3Qgb25lIGl0ZW0gaWYgcG9zc2libGUuXG4gICAgLy8gSWYgdGhlIHVwc3RyZWFtIHNvdXJjZSBpcyBleGhhdXN0ZWQsIEFORCB0aGVyZSBhcmUgbm8gaXRlbXMgbGVmdCBpblxuICAgIC8vIHRoZSBvdXRwdXQgcXVldWUsIHRoZW4gdGhpcyBzdHJlYW0gaXMgYWxzbyBleGhhdXN0ZWQuXG4gICAgd2hpbGUgKHRoaXMub3V0cHV0UXVldWUubGVuZ3RoKCkgPT09IDApIHtcbiAgICAgIC8vIFRPRE8oc29lcmdlbCk6IGNvbnNpZGVyIHBhcmFsbGVsIHJlYWRzLlxuICAgICAgaWYgKCFhd2FpdCB0aGlzLnB1bXAoKSkge1xuICAgICAgICByZXR1cm4ge3ZhbHVlOiBudWxsLCBkb25lOiB0cnVlfTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHt2YWx1ZTogdGhpcy5vdXRwdXRRdWV1ZS5zaGlmdCgpLCBkb25lOiBmYWxzZX07XG4gIH1cbn1cbmNsYXNzIEZsYXRtYXBJdGVyYXRvcjxJLCBPPiBleHRlbmRzIE9uZVRvTWFueUl0ZXJhdG9yPE8+IHtcbiAgY29uc3RydWN0b3IoXG4gICAgICBwcm90ZWN0ZWQgdXBzdHJlYW06IExhenlJdGVyYXRvcjxJPixcbiAgICAgIHByb3RlY3RlZCB0cmFuc2Zvcm06ICh2YWx1ZTogSSkgPT4gT1tdKSB7XG4gICAgc3VwZXIoKTtcbiAgfVxuXG4gIHN1bW1hcnkoKSB7XG4gICAgcmV0dXJuIGAke3RoaXMudXBzdHJlYW0uc3VtbWFyeSgpfSAtPiBGbGF0bWFwYDtcbiAgfVxuXG4gIGFzeW5jIHB1bXAoKTogUHJvbWlzZTxib29sZWFuPiB7XG4gICAgY29uc3QgaXRlbSA9IGF3YWl0IHRoaXMudXBzdHJlYW0ubmV4dCgpO1xuICAgIGlmIChpdGVtLmRvbmUpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gICAgY29uc3QgaW5wdXRUZW5zb3JzID0gdGYudGVuc29yX3V0aWwuZ2V0VGVuc29yc0luQ29udGFpbmVyKGl0ZW0udmFsdWUgYXMge30pO1xuICAgIC8vIENhcmVmdWw6IHRoZSB0cmFuc2Zvcm0gbWF5IG11dGF0ZSB0aGUgaXRlbSBpbiBwbGFjZS5cbiAgICAvLyB0aGF0J3Mgd2h5IHdlIGhhdmUgdG8gcmVtZW1iZXIgdGhlIGlucHV0IFRlbnNvcnMgYWJvdmUsIGFuZCB0aGVuXG4gICAgLy8gYmVsb3cgZGlzcG9zZSBvbmx5IHRob3NlIHRoYXQgd2VyZSBub3QgcGFzc2VkIHRocm91Z2ggdG8gdGhlIG91dHB1dC5cbiAgICAvLyBOb3RlIHRvbyB0aGF0IHRoZSB0cmFuc2Zvcm0gZnVuY3Rpb24gaXMgcmVzcG9uc2libGUgZm9yIHRpZHlpbmcgYW55XG4gICAgLy8gaW50ZXJtZWRpYXRlIFRlbnNvcnMuICBIZXJlIHdlIGFyZSBjb25jZXJuZWQgb25seSBhYm91dCB0aGUgaW5wdXRzLlxuICAgIGNvbnN0IG1hcHBlZEFycmF5ID0gdGhpcy50cmFuc2Zvcm0oaXRlbS52YWx1ZSk7XG4gICAgY29uc3Qgb3V0cHV0VGVuc29ycyA9XG4gICAgICAgIHRmLnRlbnNvcl91dGlsLmdldFRlbnNvcnNJbkNvbnRhaW5lcihtYXBwZWRBcnJheSBhcyB7fSk7XG4gICAgdGhpcy5vdXRwdXRRdWV1ZS5wdXNoQWxsKG1hcHBlZEFycmF5KTtcblxuICAgIC8vIFRPRE8oc29lcmdlbCkgZmFzdGVyIGludGVyc2VjdGlvbiwgYW5kIGRlZHVwbGljYXRlIG91dHB1dFRlbnNvcnNcbiAgICAvLyBUT0RPKHNvZXJnZWwpIG1vdmUgdG8gdGYuZGlzcG9zZUV4Y2VwdChpbiwgb3V0KT9cbiAgICBmb3IgKGNvbnN0IHQgb2YgaW5wdXRUZW5zb3JzKSB7XG4gICAgICBpZiAoIXRmLnRlbnNvcl91dGlsLmlzVGVuc29ySW5MaXN0KHQsIG91dHB1dFRlbnNvcnMpKSB7XG4gICAgICAgIHQuZGlzcG9zZSgpO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB0cnVlO1xuICB9XG59XG5cbi8qKlxuICogUHJvdmlkZXMgYSBgTGF6eUl0ZXJhdG9yYCB0aGF0IGNvbmNhdGVuYXRlcyBhIHN0cmVhbSBvZiB1bmRlcmx5aW5nXG4gKiBzdHJlYW1zLlxuICpcbiAqIERvaW5nIHRoaXMgaW4gYSBjb25jdXJyZW5jeS1zYWZlIHdheSByZXF1aXJlcyBzb21lIHRyaWNrZXJ5LiAgSW5cbiAqIHBhcnRpY3VsYXIsIHdlIHdhbnQgdGhpcyBzdHJlYW0gdG8gcmV0dXJuIHRoZSBlbGVtZW50cyBmcm9tIHRoZVxuICogdW5kZXJseWluZyBzdHJlYW1zIGluIHRoZSBjb3JyZWN0IG9yZGVyIGFjY29yZGluZyB0byB3aGVuIG5leHQoKSB3YXNcbiAqIGNhbGxlZCwgZXZlbiBpZiB0aGUgcmVzdWx0aW5nIFByb21pc2VzIHJlc29sdmUgaW4gYSBkaWZmZXJlbnQgb3JkZXIuXG4gKi9cbmV4cG9ydCBjbGFzcyBDaGFpbmVkSXRlcmF0b3I8VD4gZXh0ZW5kcyBMYXp5SXRlcmF0b3I8VD4ge1xuICAvLyBTdHJpY3QgUHJvbWlzZSBleGVjdXRpb24gb3JkZXI6XG4gIC8vIGEgbmV4dCgpIGNhbGwgbWF5IG5vdCBldmVuIGJlZ2luIHVudGlsIHRoZSBwcmV2aW91cyBvbmUgY29tcGxldGVzLlxuICBwcml2YXRlIGxhc3RSZWFkOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PiA9IG51bGw7XG5cbiAgLy8gTG9jYWwgc3RhdGUgdGhhdCBzaG91bGQgbm90IGJlIGNsb2JiZXJlZCBieSBvdXQtb2Ytb3JkZXIgZXhlY3V0aW9uLlxuICBwcml2YXRlIGl0ZXJhdG9yOiBMYXp5SXRlcmF0b3I8VD4gPSBudWxsO1xuICBwcml2YXRlIG1vcmVJdGVyYXRvcnM6IExhenlJdGVyYXRvcjxMYXp5SXRlcmF0b3I8VD4+O1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgaXRlcmF0b3JzOiBMYXp5SXRlcmF0b3I8TGF6eUl0ZXJhdG9yPFQ+PixcbiAgICAgIHByaXZhdGUgcmVhZG9ubHkgYmFzZUVycm9ySGFuZGxlcj86IChlOiBFcnJvcikgPT4gYm9vbGVhbikge1xuICAgIHN1cGVyKCk7XG4gICAgdGhpcy5tb3JlSXRlcmF0b3JzID0gaXRlcmF0b3JzO1xuICB9XG5cbiAgc3VtbWFyeSgpIHtcbiAgICBjb25zdCB1cHN0cmVhbVN1bW1hcmllcyA9ICdUT0RPOiBmaWxsIGluIHVwc3RyZWFtIG9mIGNoYWluZWQgc3VtbWFyaWVzJztcbiAgICByZXR1cm4gYCR7dXBzdHJlYW1TdW1tYXJpZXN9IC0+IENoYWluZWRgO1xuICB9XG5cbiAgYXN5bmMgbmV4dCgpOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PiB7XG4gICAgdGhpcy5sYXN0UmVhZCA9IHRoaXMucmVhZEZyb21DaGFpbih0aGlzLmxhc3RSZWFkKTtcbiAgICByZXR1cm4gdGhpcy5sYXN0UmVhZDtcbiAgfVxuXG4gIHByaXZhdGUgYXN5bmMgcmVhZEZyb21DaGFpbihsYXN0UmVhZDogUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxUPj4pOlxuICAgICAgUHJvbWlzZTxJdGVyYXRvclJlc3VsdDxUPj4ge1xuICAgIC8vIE11c3QgYXdhaXQgb24gdGhlIHByZXZpb3VzIHJlYWQgc2luY2UgdGhlIHByZXZpb3VzIHJlYWQgbWF5IGhhdmUgYWR2YW5jZWRcbiAgICAvLyB0aGUgc3RyZWFtIG9mIHN0cmVhbXMsIGZyb20gd2hpY2ggd2UgbmVlZCB0byByZWFkLlxuICAgIC8vIFRoaXMgaXMgdW5mb3J0dW5hdGUgc2luY2Ugd2UgY2FuJ3QgcGFyYWxsZWxpemUgcmVhZHMuIFdoaWNoIG1lYW5zXG4gICAgLy8gcHJlZmV0Y2hpbmcgb2YgY2hhaW5lZCBzdHJlYW1zIGlzIGEgbm8tb3AuXG4gICAgLy8gT25lIHNvbHV0aW9uIGlzIHRvIHByZWZldGNoIGltbWVkaWF0ZWx5IHVwc3RyZWFtIG9mIHRoaXMuXG4gICAgYXdhaXQgbGFzdFJlYWQ7XG4gICAgaWYgKHRoaXMuaXRlcmF0b3IgPT0gbnVsbCkge1xuICAgICAgY29uc3QgaXRlcmF0b3JSZXN1bHQgPSBhd2FpdCB0aGlzLm1vcmVJdGVyYXRvcnMubmV4dCgpO1xuICAgICAgaWYgKGl0ZXJhdG9yUmVzdWx0LmRvbmUpIHtcbiAgICAgICAgLy8gTm8gbW9yZSBzdHJlYW1zIHRvIHN0cmVhbSBmcm9tLlxuICAgICAgICByZXR1cm4ge3ZhbHVlOiBudWxsLCBkb25lOiB0cnVlfTtcbiAgICAgIH1cbiAgICAgIHRoaXMuaXRlcmF0b3IgPSBpdGVyYXRvclJlc3VsdC52YWx1ZTtcbiAgICAgIGlmICh0aGlzLmJhc2VFcnJvckhhbmRsZXIgIT0gbnVsbCkge1xuICAgICAgICB0aGlzLml0ZXJhdG9yID0gdGhpcy5pdGVyYXRvci5oYW5kbGVFcnJvcnModGhpcy5iYXNlRXJyb3JIYW5kbGVyKTtcbiAgICAgIH1cbiAgICB9XG4gICAgY29uc3QgaXRlbVJlc3VsdCA9IGF3YWl0IHRoaXMuaXRlcmF0b3IubmV4dCgpO1xuICAgIGlmIChpdGVtUmVzdWx0LmRvbmUpIHtcbiAgICAgIHRoaXMuaXRlcmF0b3IgPSBudWxsO1xuICAgICAgcmV0dXJuIHRoaXMucmVhZEZyb21DaGFpbihsYXN0UmVhZCk7XG4gICAgfVxuICAgIHJldHVybiBpdGVtUmVzdWx0O1xuICB9XG59XG5cbmV4cG9ydCBlbnVtIFppcE1pc21hdGNoTW9kZSB7XG4gIEZBSUwsICAgICAgLy8gcmVxdWlyZSB6aXBwZWQgc3RyZWFtcyB0byBoYXZlIHRoZSBzYW1lIGxlbmd0aFxuICBTSE9SVEVTVCwgIC8vIHRlcm1pbmF0ZSB6aXAgd2hlbiB0aGUgZmlyc3Qgc3RyZWFtIGlzIGV4aGF1c3RlZFxuICBMT05HRVNUICAgIC8vIHVzZSBudWxscyBmb3IgZXhoYXVzdGVkIHN0cmVhbXM7IHVzZSB1cCB0aGUgbG9uZ2VzdCBzdHJlYW0uXG59XG5cbi8qKlxuICogUHJvdmlkZXMgYSBgTGF6eUl0ZXJhdG9yYCB0aGF0IHppcHMgdG9nZXRoZXIgYW4gYXJyYXksIGRpY3QsIG9yIG5lc3RlZFxuICogc3RydWN0dXJlIG9mIGBMYXp5SXRlcmF0b3JgcyAoYW5kIHBlcmhhcHMgYWRkaXRpb25hbCBjb25zdGFudHMpLlxuICpcbiAqIFRoZSB1bmRlcmx5aW5nIHN0cmVhbXMgbXVzdCBwcm92aWRlIGVsZW1lbnRzIGluIGEgY29uc2lzdGVudCBvcmRlciBzdWNoXG4gKiB0aGF0IHRoZXkgY29ycmVzcG9uZC5cbiAqXG4gKiBUeXBpY2FsbHksIHRoZSB1bmRlcmx5aW5nIHN0cmVhbXMgc2hvdWxkIGhhdmUgdGhlIHNhbWUgbnVtYmVyIG9mXG4gKiBlbGVtZW50cy4gSWYgdGhleSBkbyBub3QsIHRoZSBiZWhhdmlvciBpcyBkZXRlcm1pbmVkIGJ5IHRoZVxuICogYG1pc21hdGNoTW9kZWAgYXJndW1lbnQuXG4gKlxuICogVGhlIG5lc3RlZCBzdHJ1Y3R1cmUgb2YgdGhlIGBpdGVyYXRvcnNgIGFyZ3VtZW50IGRldGVybWluZXMgdGhlXG4gKiBzdHJ1Y3R1cmUgb2YgZWxlbWVudHMgaW4gdGhlIHJlc3VsdGluZyBpdGVyYXRvci5cbiAqXG4gKiBEb2luZyB0aGlzIGluIGEgY29uY3VycmVuY3ktc2FmZSB3YXkgcmVxdWlyZXMgc29tZSB0cmlja2VyeS4gIEluXG4gKiBwYXJ0aWN1bGFyLCB3ZSB3YW50IHRoaXMgc3RyZWFtIHRvIHJldHVybiB0aGUgZWxlbWVudHMgZnJvbSB0aGVcbiAqIHVuZGVybHlpbmcgc3RyZWFtcyBpbiB0aGUgY29ycmVjdCBvcmRlciBhY2NvcmRpbmcgdG8gd2hlbiBuZXh0KCkgd2FzXG4gKiBjYWxsZWQsIGV2ZW4gaWYgdGhlIHJlc3VsdGluZyBQcm9taXNlcyByZXNvbHZlIGluIGEgZGlmZmVyZW50IG9yZGVyLlxuICpcbiAqIEBwYXJhbSBpdGVyYXRvcnM6IEFuIGFycmF5IG9yIG9iamVjdCBjb250YWluaW5nIExhenlJdGVyYXRvcnMgYXQgdGhlXG4gKiBsZWF2ZXMuXG4gKiBAcGFyYW0gbWlzbWF0Y2hNb2RlOiBEZXRlcm1pbmVzIHdoYXQgdG8gZG8gd2hlbiBvbmUgdW5kZXJseWluZyBpdGVyYXRvclxuICogaXMgZXhoYXVzdGVkIGJlZm9yZSB0aGUgb3RoZXJzLiAgYFppcE1pc21hdGNoTW9kZS5GQUlMYCAodGhlIGRlZmF1bHQpXG4gKiBjYXVzZXMgYW4gZXJyb3IgdG8gYmUgdGhyb3duIGluIHRoaXMgY2FzZS4gIGBaaXBNaXNtYXRjaE1vZGUuU0hPUlRFU1RgXG4gKiBjYXVzZXMgdGhlIHppcHBlZCBpdGVyYXRvciB0byB0ZXJtaW5hdGUgd2l0aCB0aGUgZnVyc3QgdW5kZXJseWluZ1xuICogc3RyZWFtcywgc28gZWxlbWVudHMgcmVtYWluaW5nIG9uIHRoZSBsb25nZXIgc3RyZWFtcyBhcmUgaWdub3JlZC5cbiAqIGBaaXBNaXNtYXRjaE1vZGUuTE9OR0VTVGAgY2F1c2VzIHRoZSB6aXBwZWQgc3RyZWFtIHRvIGNvbnRpbnVlLCBmaWxsaW5nXG4gKiBpbiBudWxscyBmb3IgdGhlIGV4aGF1c3RlZCBzdHJlYW1zLCB1bnRpbCBhbGwgc3RyZWFtcyBhcmUgZXhoYXVzdGVkLlxuICovXG5jbGFzcyBaaXBJdGVyYXRvcjxPIGV4dGVuZHMgdGYuVGVuc29yQ29udGFpbmVyPiBleHRlbmRzIExhenlJdGVyYXRvcjxPPiB7XG4gIHByaXZhdGUgY291bnQgPSAwO1xuICBwcml2YXRlIGN1cnJlbnRQcm9taXNlOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PE8+PiA9IG51bGw7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBwcm90ZWN0ZWQgcmVhZG9ubHkgaXRlcmF0b3JzOiBJdGVyYXRvckNvbnRhaW5lcixcbiAgICAgIHByb3RlY3RlZCByZWFkb25seSBtaXNtYXRjaE1vZGU6IFppcE1pc21hdGNoTW9kZSA9IFppcE1pc21hdGNoTW9kZS5GQUlMKSB7XG4gICAgc3VwZXIoKTtcbiAgfVxuXG4gIHN1bW1hcnkoKSB7XG4gICAgY29uc3QgdXBzdHJlYW1TdW1tYXJpZXMgPSAnVE9ETzogZmlsbCBpbiB1cHN0cmVhbSBvZiB6aXAgc3VtbWFyaWVzJztcbiAgICByZXR1cm4gYHske3Vwc3RyZWFtU3VtbWFyaWVzfX0gLT4gWmlwYDtcbiAgfVxuXG4gIHByaXZhdGUgYXN5bmMgbmV4dFN0YXRlKGFmdGVyU3RhdGU6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8Tz4+KTpcbiAgICAgIFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8Tz4+IHtcbiAgICAvLyBUaGlzIGNoYWluaW5nIGVuc3VyZXMgdGhhdCB0aGUgdW5kZXJseWluZyBuZXh0KCkgYXJlIG5vdCBldmVuIGNhbGxlZFxuICAgIC8vIGJlZm9yZSB0aGUgcHJldmlvdXMgb25lcyBoYXZlIHJlc29sdmVkLlxuICAgIGF3YWl0IGFmdGVyU3RhdGU7XG5cbiAgICAvLyBDb2xsZWN0IHVuZGVybHlpbmcgaXRlcmF0b3IgXCJkb25lXCIgc2lnbmFscyBhcyBhIHNpZGUgZWZmZWN0IGluXG4gICAgLy8gZ2V0TmV4dCgpXG4gICAgbGV0IG51bUl0ZXJhdG9ycyA9IDA7XG4gICAgbGV0IGl0ZXJhdG9yc0RvbmUgPSAwO1xuXG4gICAgZnVuY3Rpb24gZ2V0TmV4dChjb250YWluZXI6IEl0ZXJhdG9yQ29udGFpbmVyKTogRGVlcE1hcEFzeW5jUmVzdWx0IHtcbiAgICAgIGlmIChjb250YWluZXIgaW5zdGFuY2VvZiBMYXp5SXRlcmF0b3IpIHtcbiAgICAgICAgY29uc3QgcmVzdWx0ID0gY29udGFpbmVyLm5leHQoKTtcbiAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICB2YWx1ZTogcmVzdWx0LnRoZW4oeCA9PiB7XG4gICAgICAgICAgICBudW1JdGVyYXRvcnMrKztcbiAgICAgICAgICAgIGlmICh4LmRvbmUpIHtcbiAgICAgICAgICAgICAgaXRlcmF0b3JzRG9uZSsrO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIHgudmFsdWU7XG4gICAgICAgICAgfSksXG4gICAgICAgICAgcmVjdXJzZTogZmFsc2VcbiAgICAgICAgfTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiB7dmFsdWU6IG51bGwsIHJlY3Vyc2U6IHRydWV9O1xuICAgICAgfVxuICAgIH1cblxuICAgIGNvbnN0IG1hcHBlZDogTyA9IGF3YWl0IGRlZXBNYXBBbmRBd2FpdEFsbCh0aGlzLml0ZXJhdG9ycywgZ2V0TmV4dCk7XG5cbiAgICBpZiAobnVtSXRlcmF0b3JzID09PSBpdGVyYXRvcnNEb25lKSB7XG4gICAgICAvLyBUaGUgc3RyZWFtcyBoYXZlIGFsbCBlbmRlZC5cbiAgICAgIHJldHVybiB7dmFsdWU6IG51bGwsIGRvbmU6IHRydWV9O1xuICAgIH1cbiAgICBpZiAoaXRlcmF0b3JzRG9uZSA+IDApIHtcbiAgICAgIHN3aXRjaCAodGhpcy5taXNtYXRjaE1vZGUpIHtcbiAgICAgICAgY2FzZSBaaXBNaXNtYXRjaE1vZGUuRkFJTDpcbiAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAgICdaaXBwZWQgc3RyZWFtcyBzaG91bGQgaGF2ZSB0aGUgc2FtZSBsZW5ndGguICcgK1xuICAgICAgICAgICAgICBgTWlzbWF0Y2hlZCBhdCBlbGVtZW50ICR7dGhpcy5jb3VudH0uYCk7XG4gICAgICAgIGNhc2UgWmlwTWlzbWF0Y2hNb2RlLlNIT1JURVNUOlxuICAgICAgICAgIHJldHVybiB7dmFsdWU6IG51bGwsIGRvbmU6IHRydWV9O1xuICAgICAgICBjYXNlIFppcE1pc21hdGNoTW9kZS5MT05HRVNUOlxuICAgICAgICBkZWZhdWx0OlxuICAgICAgICAgIC8vIENvbnRpbnVlLiAgVGhlIGV4aGF1c3RlZCBzdHJlYW1zIGFscmVhZHkgcHJvZHVjZWQgdmFsdWU6IG51bGwuXG4gICAgICB9XG4gICAgfVxuXG4gICAgdGhpcy5jb3VudCsrO1xuICAgIHJldHVybiB7dmFsdWU6IG1hcHBlZCwgZG9uZTogZmFsc2V9O1xuICB9XG5cbiAgYXN5bmMgbmV4dCgpOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PE8+PiB7XG4gICAgdGhpcy5jdXJyZW50UHJvbWlzZSA9IHRoaXMubmV4dFN0YXRlKHRoaXMuY3VycmVudFByb21pc2UpO1xuICAgIHJldHVybiB0aGlzLmN1cnJlbnRQcm9taXNlO1xuICB9XG59XG5cbi8vIEl0ZXJhdG9ycyB0aGF0IG1haW50YWluIGEgcmluZyBidWZmZXIgb2YgcGVuZGluZyBwcm9taXNlc1xuLy8gPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuXG4vKipcbiAqIEEgc3RyZWFtIHRoYXQgcHJlZmV0Y2hlcyBhIGdpdmVuIG51bWJlciBvZiBpdGVtcyBmcm9tIGFuIHVwc3RyZWFtIHNvdXJjZSxcbiAqIHJldHVybmluZyB0aGVtIGluIEZJRk8gb3JkZXIuXG4gKlxuICogTm90ZSB0aGlzIHByZWZldGNoZXMgUHJvbWlzZXMsIGJ1dCBtYWtlcyBubyBndWFyYW50ZWVzIGFib3V0IHdoZW4gdGhvc2VcbiAqIFByb21pc2VzIHJlc29sdmUuXG4gKi9cbmV4cG9ydCBjbGFzcyBQcmVmZXRjaEl0ZXJhdG9yPFQ+IGV4dGVuZHMgTGF6eUl0ZXJhdG9yPFQ+IHtcbiAgcHJvdGVjdGVkIGJ1ZmZlcjogUmluZ0J1ZmZlcjxQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+Pj47XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBwcm90ZWN0ZWQgdXBzdHJlYW06IExhenlJdGVyYXRvcjxUPiwgcHJvdGVjdGVkIGJ1ZmZlclNpemU6IG51bWJlcikge1xuICAgIHN1cGVyKCk7XG4gICAgdGhpcy5idWZmZXIgPSBuZXcgUmluZ0J1ZmZlcjxQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+Pj4oYnVmZmVyU2l6ZSk7XG4gIH1cblxuICBzdW1tYXJ5KCkge1xuICAgIHJldHVybiBgJHt0aGlzLnVwc3RyZWFtLnN1bW1hcnkoKX0gLT4gUHJlZmV0Y2hgO1xuICB9XG5cbiAgLyoqXG4gICAqIFJlZmlsbCB0aGUgcHJlZmV0Y2ggYnVmZmVyLiAgUmV0dXJucyBvbmx5IGFmdGVyIHRoZSBidWZmZXIgaXMgZnVsbCwgb3JcbiAgICogdGhlIHVwc3RyZWFtIHNvdXJjZSBpcyBleGhhdXN0ZWQuXG4gICAqL1xuICBwcm90ZWN0ZWQgcmVmaWxsKCkge1xuICAgIHdoaWxlICghdGhpcy5idWZmZXIuaXNGdWxsKCkpIHtcbiAgICAgIGNvbnN0IHYgPSB0aGlzLnVwc3RyZWFtLm5leHQoKTtcbiAgICAgIHRoaXMuYnVmZmVyLnB1c2godik7XG4gICAgfVxuICB9XG5cbiAgbmV4dCgpOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PiB7XG4gICAgdGhpcy5yZWZpbGwoKTtcbiAgICAvLyBUaGlzIHNoaWZ0IHdpbGwgbmV2ZXIgdGhyb3cgYW4gZXJyb3IgYmVjYXVzZSB0aGUgYnVmZmVyIGlzIGFsd2F5c1xuICAgIC8vIGZ1bGwgYWZ0ZXIgYSByZWZpbGwuIElmIHRoZSBzdHJlYW0gaXMgZXhoYXVzdGVkLCB0aGUgYnVmZmVyIHdpbGwgYmVcbiAgICAvLyBmdWxsIG9mIFByb21pc2VzIHRoYXQgd2lsbCByZXNvbHZlIHRvIHRoZSBlbmQtb2Ytc3RyZWFtIHNpZ25hbC5cbiAgICByZXR1cm4gdGhpcy5idWZmZXIuc2hpZnQoKTtcbiAgfVxufVxuXG4vKipcbiAqIEEgc3RyZWFtIHRoYXQgcGVyZm9ybXMgYSBzbGlkaW5nLXdpbmRvdyByYW5kb20gc2h1ZmZsZSBvbiBhbiB1cHN0cmVhbVxuICogc291cmNlLiBUaGlzIGlzIGxpa2UgYSBgUHJlZmV0Y2hJdGVyYXRvcmAgZXhjZXB0IHRoYXQgdGhlIGl0ZW1zIGFyZVxuICogcmV0dXJuZWQgaW4gcmFuZG9taXplZCBvcmRlci4gIE1peGluZyBuYXR1cmFsbHkgaW1wcm92ZXMgYXMgdGhlIGJ1ZmZlclxuICogc2l6ZSBpbmNyZWFzZXMuXG4gKi9cbmV4cG9ydCBjbGFzcyBTaHVmZmxlSXRlcmF0b3I8VD4gZXh0ZW5kcyBQcmVmZXRjaEl0ZXJhdG9yPFQ+IHtcbiAgcHJpdmF0ZSByZWFkb25seSByYW5kb206IHNlZWRyYW5kb20ucHJuZztcblxuICAvLyBTdHJpY3QgUHJvbWlzZSBleGVjdXRpb24gb3JkZXI6XG4gIC8vIGEgbmV4dCgpIGNhbGwgbWF5IG5vdCBldmVuIGJlZ2luIHVudGlsIHRoZSBwcmV2aW91cyBvbmUgY29tcGxldGVzLlxuICBwcml2YXRlIGxhc3RSZWFkOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PjtcblxuICAvLyBMb2NhbCBzdGF0ZSB0aGF0IHNob3VsZCBub3QgYmUgY2xvYmJlcmVkIGJ5IG91dC1vZi1vcmRlciBleGVjdXRpb24uXG4gIHByaXZhdGUgdXBzdHJlYW1FeGhhdXN0ZWQgPSBmYWxzZTtcblxuICBjb25zdHJ1Y3RvcihcbiAgICBwcm90ZWN0ZWQgb3ZlcnJpZGUgdXBzdHJlYW06IExhenlJdGVyYXRvcjxUPiwgcHJvdGVjdGVkIHdpbmRvd1NpemU6IG51bWJlcixcbiAgICAgIHNlZWQ/OiBzdHJpbmcpIHtcbiAgICBzdXBlcih1cHN0cmVhbSwgd2luZG93U2l6ZSk7XG4gICAgdGhpcy5yYW5kb20gPSBzZWVkcmFuZG9tLmFsZWEoc2VlZCB8fCB0Zi51dGlsLm5vdygpLnRvU3RyaW5nKCkpO1xuICAgIHRoaXMubGFzdFJlYWQgPSBQcm9taXNlLnJlc29sdmUoe3ZhbHVlOiBudWxsLCBkb25lOiBmYWxzZX0pO1xuICB9XG5cbiAgb3ZlcnJpZGUgYXN5bmMgbmV4dCgpOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PiB7XG4gICAgLy8gVGhpcyBzZXRzIHRoaXMubGFzdFJlYWQgdG8gYSBuZXcgUHJvbWlzZSByaWdodCBhd2F5LCBhcyBvcHBvc2VkIHRvXG4gICAgLy8gc2F5aW5nIGBhd2FpdCB0aGlzLmxhc3RSZWFkOyB0aGlzLmxhc3RSZWFkID0gdGhpcy5zZXJpYWxOZXh0KCk7YCB3aGljaFxuICAgIC8vIHdvdWxkIG5vdCB3b3JrIGJlY2F1c2UgdGhpcy5uZXh0UmVhZCB3b3VsZCBiZSB1cGRhdGVkIG9ubHkgYWZ0ZXIgdGhlXG4gICAgLy8gcHJvbWlzZSByZXNvbHZlcy5cbiAgICB0aGlzLmxhc3RSZWFkID0gdGhpcy5sYXN0UmVhZC50aGVuKCgpID0+IHRoaXMuc2VyaWFsTmV4dCgpKTtcbiAgICByZXR1cm4gdGhpcy5sYXN0UmVhZDtcbiAgfVxuXG4gIHByaXZhdGUgcmFuZG9tSW50KG1heDogbnVtYmVyKSB7XG4gICAgcmV0dXJuIE1hdGguZmxvb3IodGhpcy5yYW5kb20oKSAqIG1heCk7XG4gIH1cblxuICBwcm90ZWN0ZWQgY2hvb3NlSW5kZXgoKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5yYW5kb21JbnQodGhpcy5idWZmZXIubGVuZ3RoKCkpO1xuICB9XG5cbiAgYXN5bmMgc2VyaWFsTmV4dCgpOiBQcm9taXNlPEl0ZXJhdG9yUmVzdWx0PFQ+PiB7XG4gICAgLy8gVE9ETyhzb2VyZ2VsKTogY29uc2lkZXIgcGVyZm9ybWFuY2VcbiAgICBpZiAoIXRoaXMudXBzdHJlYW1FeGhhdXN0ZWQpIHtcbiAgICAgIHRoaXMucmVmaWxsKCk7XG4gICAgfVxuICAgIHdoaWxlICghdGhpcy5idWZmZXIuaXNFbXB0eSgpKSB7XG4gICAgICBjb25zdCBjaG9zZW5JbmRleCA9IHRoaXMuY2hvb3NlSW5kZXgoKTtcbiAgICAgIGNvbnN0IHJlc3VsdCA9IGF3YWl0IHRoaXMuYnVmZmVyLnNodWZmbGVFeGNpc2UoY2hvc2VuSW5kZXgpO1xuICAgICAgaWYgKHJlc3VsdC5kb25lKSB7XG4gICAgICAgIHRoaXMudXBzdHJlYW1FeGhhdXN0ZWQgPSB0cnVlO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhpcy5yZWZpbGwoKTtcbiAgICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHt2YWx1ZTogbnVsbCwgZG9uZTogdHJ1ZX07XG4gIH1cbn1cbiJdfQ==