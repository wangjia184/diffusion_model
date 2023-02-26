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
import { iteratorFromConcatenated, iteratorFromFunction, iteratorFromItems, iteratorFromZipped, ZipMismatchMode } from './iterators/lazy_iterator';
import { canTensorify, deepMapAndAwaitAll, isIterable } from './util/deep_map';
// TODO(soergel): consider vectorized operations within the pipeline.
/**
 * Represents a potentially large list of independent data elements (typically
 * 'samples' or 'examples').
 *
 * A 'data example' may be a primitive, an array, a map from string keys to
 * values, or any nested structure of these.
 *
 * A `Dataset` represents an ordered collection of elements, together with a
 * chain of transformations to be performed on those elements. Each
 * transformation is a method of `Dataset` that returns another `Dataset`, so
 * these may be chained, e.g.
 * `const processedDataset = rawDataset.filter(...).map(...).batch(...)`.
 *
 * Data loading and transformation is done in a lazy, streaming fashion.  The
 * dataset may be iterated over multiple times; each iteration starts the data
 * loading anew and recapitulates the transformations.
 *
 * A `Dataset` is typically processed as a stream of unbatched examples -- i.e.,
 * its transformations are applied one example at a time. Batching produces a
 * new `Dataset` where each element is a batch. Batching should usually come
 * last in a pipeline, because data transformations are easier to express on a
 * per-example basis than on a per-batch basis.
 *
 * The following code examples are calling `await dataset.forEachAsync(...)` to
 * iterate once over the entire dataset in order to print out the data.
 *
 * @doc {heading: 'Data', subheading: 'Classes', namespace: 'data'}
 */
export class Dataset {
    constructor() {
        this.size = null;
    }
    // TODO(soergel): Make Datasets report whether repeated iterator() calls
    // produce the same result (e.g., reading from a file) or different results
    // (e.g., from the webcam).  Currently we don't make this distinction but it
    // could be important for the user to know.
    // abstract isDeterministic(): boolean;
    /**
     * Groups elements into batches.
     *
     * It is assumed that each of the incoming dataset elements has the same
     * structure -- i.e. the same set of keys at each location in an object
     * hierarchy.  For each key, the resulting `Dataset` provides a batched
     * element collecting all of the incoming values for that key.
     *
     *  * Incoming primitives are grouped into a 1-D Tensor.
     *  * Incoming Tensors are grouped into a new Tensor where the 0th axis is
     *    the batch dimension.
     *  * Incoming arrays are converted to Tensor and then batched.
     *  * A nested array is interpreted as an n-D Tensor, so the batched result
     *    has n+1 dimensions.
     *  * An array that cannot be converted to Tensor produces an error.
     *
     * If an array should not be batched as a unit, it should first be converted
     * to an object with integer keys.
     *
     * Here are a few examples:
     *
     * Batch a dataset of numbers:
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6, 7, 8]).batch(4);
     * await a.forEachAsync(e => e.print());
     * ```
     *
     * Batch a dataset of arrays:
     * ```js
     * const b = tf.data.array([[1], [2], [3], [4], [5], [6], [7], [8]]).batch(4);
     * await b.forEachAsync(e => e.print());
     * ```
     *
     * Batch a dataset of objects:
     * ```js
     * const c = tf.data.array([{a: 1, b: 11}, {a: 2, b: 12}, {a: 3, b: 13},
     *   {a: 4, b: 14}, {a: 5, b: 15}, {a: 6, b: 16}, {a: 7, b: 17},
     *   {a: 8, b: 18}]).batch(4);
     * await c.forEachAsync(e => {
     *   console.log('{');
     *   for(var key in e) {
     *     console.log(key+':');
     *     e[key].print();
     *   }
     *   console.log('}');
     * })
     * ```
     *
     * @param batchSize The number of elements desired per batch.
     * @param smallLastBatch Whether to emit the final batch when it has fewer
     *   than batchSize elements. Default true.
     * @returns A `Dataset`, from which a stream of batches can be obtained.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    batch(batchSize, smallLastBatch = true) {
        const base = this;
        tf.util.assert(batchSize > 0, () => `batchSize needs to be positive, but it is
      ${batchSize}`);
        let size;
        if (this.size === Infinity || this.size == null) {
            // If the size of this dataset is infinity or null, the new size keeps the
            // same.
            size = this.size;
        }
        else if (smallLastBatch) {
            // If the size of this dataset is known and include small last batch, the
            // new size is full batch count plus last batch.
            size = Math.ceil(this.size / batchSize);
        }
        else {
            // If the size of this dataset is known and not include small last batch,
            // the new size is full batch count.
            size = Math.floor(this.size / batchSize);
        }
        return datasetFromIteratorFn(async () => {
            return (await base.iterator())
                .columnMajorBatch(batchSize, smallLastBatch, deepBatchConcat);
        }, size);
    }
    /**
     * Concatenates this `Dataset` with another.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]);
     * const b = tf.data.array([4, 5, 6]);
     * const c = a.concatenate(b);
     * await c.forEachAsync(e => console.log(e));
     * ```
     *
     * @param dataset A `Dataset` to be concatenated onto this one.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    concatenate(dataset) {
        const base = this;
        let size;
        if (this.size === Infinity || dataset.size === Infinity) {
            // If the size of any of these two dataset is infinity, new size is
            // infinity.
            size = Infinity;
        }
        else if (this.size != null && dataset.size != null) {
            // If the size of both datasets are known and not infinity, new size is
            // sum the size of these two datasets.
            size = this.size + dataset.size;
        }
        else {
            // If neither of these two datasets has infinite size and any of these two
            // datasets' size is null, the new size is null.
            size = null;
        }
        return datasetFromIteratorFn(async () => (await base.iterator()).concatenate(await dataset.iterator()), size);
    }
    /**
     * Filters this dataset according to `predicate`.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
     *   .filter(x => x%2 === 0);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param predicate A function mapping a dataset element to a boolean or a
     * `Promise` for one.
     *
     * @returns A `Dataset` of elements for which the predicate was true.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    filter(predicate) {
        const base = this;
        let size;
        if (this.size === Infinity) {
            // If the size of this dataset is infinity, new size is infinity
            size = Infinity;
        }
        else {
            // If this dataset has limited elements, new size is null because it might
            // exhausted randomly.
            size = null;
        }
        return datasetFromIteratorFn(async () => {
            return (await base.iterator()).filter(x => tf.tidy(() => predicate(x)));
        }, size);
    }
    /**
     * Apply a function to every element of the dataset.
     *
     * After the function is applied to a dataset element, any Tensors contained
     * within that element are disposed.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param f A function to apply to each dataset element.
     * @returns A `Promise` that resolves after all elements have been processed.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    async forEachAsync(f) {
        return (await this.iterator()).forEachAsync(f);
    }
    /**
     * Maps this dataset through a 1-to-1 transform.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]).map(x => x*x);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param transform A function mapping a dataset element to a transformed
     *   dataset element.
     *
     * @returns A `Dataset` of transformed elements.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    map(transform) {
        const base = this;
        return datasetFromIteratorFn(async () => {
            return (await base.iterator()).map(x => tf.tidy(() => transform(x)));
        }, this.size);
    }
    /**
     * Maps this dataset through an async 1-to-1 transform.
     *
     * ```js
     * const a =
     *  tf.data.array([1, 2, 3]).mapAsync(x => new Promise(function(resolve){
     *    setTimeout(() => {
     *      resolve(x * x);
     *    }, Math.random()*1000 + 500);
     *  }));
     * console.log(await a.toArray());
     * ```
     *
     * @param transform A function mapping a dataset element to a `Promise` for a
     *   transformed dataset element.  This transform is responsible for disposing
     *   any intermediate `Tensor`s, i.e. by wrapping its computation in
     *   `tf.tidy()`; that cannot be automated here (as it is in the synchronous
     *   `map()` case).
     *
     * @returns A `Dataset` of transformed elements.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    mapAsync(transform) {
        const base = this;
        return datasetFromIteratorFn(async () => {
            return (await base.iterator()).mapAsync(transform);
        }, this.size);
    }
    /**
     *  Creates a `Dataset` that prefetches elements from this dataset.
     *
     * @param bufferSize: An integer specifying the number of elements to be
     *   prefetched.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    prefetch(bufferSize) {
        if (bufferSize == null) {
            throw new RangeError('`Dataset.prefetch()` requires bufferSize to be specified.');
        }
        const base = this;
        return datasetFromIteratorFn(async () => (await base.iterator()).prefetch(bufferSize), this.size);
    }
    /**
     * Repeats this dataset `count` times.
     *
     * NOTE: If this dataset is a function of global state (e.g. a random number
     * generator), then different repetitions may produce different elements.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]).repeat(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param count: (Optional) An integer, representing the number of times
     *   the dataset should be repeated. The default behavior (if `count` is
     *   `undefined` or negative) is for the dataset be repeated indefinitely.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    repeat(count) {
        const base = this;
        let size;
        if (this.size != null && count > 0) {
            // If this dataset has size and count is positive, new size is current
            // size multiply count. This also covers the case that current size is
            // infinity.
            size = this.size * count;
        }
        else if (count === 0) {
            // If count is 0, new size is 0.
            size = 0;
        }
        else if (this.size != null && (count === undefined || count < 0)) {
            // If this dataset has size and count is undefined or negative, the
            // dataset will be repeated indefinitely and new size is infinity.
            size = Infinity;
        }
        else {
            // If the size of this dataset is null, the new dataset's size is null.
            size = null;
        }
        return datasetFromIteratorFn(async () => {
            const iteratorIterator = iteratorFromFunction(async () => ({ value: await base.iterator(), done: false }));
            return iteratorFromConcatenated(iteratorIterator.take(count));
        }, size);
    }
    /**
     * Creates a `Dataset` that skips `count` initial elements from this dataset.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]).skip(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param count: The number of elements of this dataset that should be skipped
     *   to form the new dataset.  If `count` is greater than the size of this
     *   dataset, the new dataset will contain no elements.  If `count`
     *   is `undefined` or negative, skips the entire dataset.
     *
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    skip(count) {
        const base = this;
        let size;
        if (this.size != null && count >= 0 && this.size >= count) {
            // If the size of this dataset is greater than count, the new dataset's
            // size is current size minus skipped size.This also covers the case that
            // current size is infinity.
            size = this.size - count;
        }
        else if (this.size != null &&
            (this.size < count || count === undefined || count < 0)) {
            // If the size of this dataset is smaller than count, or count is
            // undefined or negative, skips the entire dataset and the new size is 0.
            size = 0;
        }
        else {
            // If the size of this dataset is null, the new dataset's size is null.
            size = null;
        }
        return datasetFromIteratorFn(async () => (await base.iterator()).skip(count), size);
    }
    /**
     * Pseudorandomly shuffles the elements of this dataset. This is done in a
     * streaming manner, by sampling from a given number of prefetched elements.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]).shuffle(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param bufferSize: An integer specifying the number of elements from this
     *   dataset from which the new dataset will sample.
     * @param seed: (Optional) An integer specifying the random seed that will
     *   be used to create the distribution.
     * @param reshuffleEachIteration: (Optional) A boolean, which if true
     *   indicates that the dataset should be pseudorandomly reshuffled each time
     *   it is iterated over. If false, elements will be returned in the same
     *   shuffled order on each iteration. (Defaults to `true`.)
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    shuffle(bufferSize, seed, reshuffleEachIteration = true) {
        if (bufferSize == null || bufferSize < 0) {
            if (this.size == null) {
                throw new RangeError('`Dataset.shuffle()` requires bufferSize to be specified.');
            }
            else {
                throw new RangeError('`Dataset.shuffle()` requires bufferSize to be specified.  ' +
                    'If your data fits in main memory (for regular JS objects), ' +
                    'and/or GPU memory (for `tf.Tensor`s), consider setting ' +
                    `bufferSize to the dataset size (${this.size} elements)`);
            }
        }
        const base = this;
        const random = seedrandom.alea(seed || tf.util.now().toString());
        return datasetFromIteratorFn(async () => {
            let seed2 = random.int32();
            if (reshuffleEachIteration) {
                seed2 += random.int32();
            }
            return (await base.iterator()).shuffle(bufferSize, seed2.toString());
        }, this.size);
    }
    /**
     * Creates a `Dataset` with at most `count` initial elements from this
     * dataset.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]).take(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param count: The number of elements of this dataset that should be taken
     *   to form the new dataset.  If `count` is `undefined` or negative, or if
     *   `count` is greater than the size of this dataset, the new dataset will
     *   contain all elements of this dataset.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    take(count) {
        const base = this;
        let size;
        if (this.size != null && this.size > count) {
            // If the size of this dataset is greater than count, the new dataset's
            // size is count.
            size = count;
        }
        else if (this.size != null && this.size <= count) {
            // If the size of this dataset is equal or smaller than count, the new
            // dataset's size is the size of this dataset.
            size = this.size;
        }
        else {
            // If the size of this dataset is null, the new dataset's size is null.
            size = null;
        }
        return datasetFromIteratorFn(async () => (await base.iterator()).take(count), size);
    }
    /**
     * Collect all elements of this dataset into an array.
     *
     * Obviously this will succeed only for small datasets that fit in memory.
     * Useful for testing and generally should be avoided if possible.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]);
     * console.log(await a.toArray());
     * ```
     *
     * @returns A Promise for an array of elements, which will resolve
     *   when a new stream has been obtained and fully consumed.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    async toArray() {
        if (this.size === Infinity) {
            throw new Error('Can not convert infinite data stream to array.');
        }
        return (await this.iterator()).toArray();
    }
    /**
     * Collect all elements of this dataset into an array with prefetching 100
     * elements. This is useful for testing, because the prefetch changes the
     * order in which the Promises are resolved along the processing pipeline.
     * This may help expose bugs where results are dependent on the order of
     * Promise resolution rather than on the logical order of the stream (i.e.,
     * due to hidden mutable state).
     *
     * @returns A Promise for an array of elements, which will resolve
     *   when a new stream has been obtained and fully consumed.
     */
    async toArrayForTest() {
        if (this.size === Infinity) {
            throw new Error('Can not convert infinite data stream to array.');
        }
        return (await this.iterator()).toArrayForTest();
    }
}
// TODO(soergel): deep sharded shuffle, where supported
Dataset.MAX_BUFFER_SIZE = 10000;
/**
 * Create a `Dataset` defined by a provided iterator() function.
 *
 * ```js
 * let i = -1;
 * const func = () =>
 *    ++i < 5 ? {value: i, done: false} : {value: null, done: true};
 * const iter = tf.data.iteratorFromFunction(func);
 * const ds = tf.data.datasetFromIteratorFn(iter);
 * await ds.forEachAsync(e => console.log(e));
 * ```
 */
export function datasetFromIteratorFn(iteratorFn, size = null) {
    return new class extends Dataset {
        constructor() {
            super(...arguments);
            this.size = size;
        }
        /*
         * Provide a new stream of elements.  Note this will also start new streams
         * from any underlying `Dataset`s.
         */
        async iterator() {
            return iteratorFn();
        }
    }();
}
/**
 * Create a `Dataset` from an array of elements.
 *
 * Create a Dataset from an array of objects:
 * ```js
 * const a = tf.data.array([{'item': 1}, {'item': 2}, {'item': 3}]);
 * await a.forEachAsync(e => console.log(e));
 * ```
 *
 * Create a Dataset from an array of numbers:
 * ```js
 * const a = tf.data.array([4, 5, 6]);
 * await a.forEachAsync(e => console.log(e));
 * ```
 * @param items An array of elements that will be parsed as items in a dataset.
 *
 * @doc {heading: 'Data', subheading: 'Creation', namespace: 'data'}
 */
export function array(items) {
    return datasetFromIteratorFn(async () => iteratorFromItems(items), items.length);
}
/**
 * Create a `Dataset` by zipping together an array, dict, or nested
 * structure of `Dataset`s (and perhaps additional constants).
 * The underlying datasets must provide elements in a consistent order such that
 * they correspond.
 *
 * The number of elements in the resulting dataset is the same as the size of
 * the smallest dataset in datasets.
 *
 * The nested structure of the `datasets` argument determines the
 * structure of elements in the resulting iterator.
 *
 * Note this means that, given an array of two datasets that produce dict
 * elements, the result is a dataset that produces elements that are arrays
 * of two dicts:
 *
 * Zip an array of datasets:
 * ```js
 * console.log('Zip two datasets of objects:');
 * const ds1 = tf.data.array([{a: 1}, {a: 2}, {a: 3}]);
 * const ds2 = tf.data.array([{b: 4}, {b: 5}, {b: 6}]);
 * const ds3 = tf.data.zip([ds1, ds2]);
 * await ds3.forEachAsync(e => console.log(JSON.stringify(e)));
 *
 * // If the goal is to merge the dicts in order to produce elements like
 * // {a: ..., b: ...}, this requires a second step such as:
 * console.log('Merge the objects:');
 * const ds4 = ds3.map(x => {return {a: x[0].a, b: x[1].b}});
 * await ds4.forEachAsync(e => console.log(e));
 * ```
 *
 * Zip a dict of datasets:
 * ```js
 * const a = tf.data.array([{a: 1}, {a: 2}, {a: 3}]);
 * const b = tf.data.array([{b: 4}, {b: 5}, {b: 6}]);
 * const c = tf.data.zip({c: a, d: b});
 * await c.forEachAsync(e => console.log(JSON.stringify(e)));
 * ```
 *
 * @doc {heading: 'Data', subheading: 'Operations', namespace: 'data'}
 */
export function zip(datasets) {
    // manually type-check the argument for JS users
    if (!isIterable(datasets)) {
        throw new Error('The argument to zip() must be an object or array.');
    }
    let size;
    if (Array.isArray(datasets)) {
        for (let i = 0; i < datasets.length; i++) {
            size = size == null ? datasets[i].size :
                Math.min(size, datasets[i].size);
        }
    }
    else if (datasets instanceof Object) {
        for (const ds in datasets) {
            size = size == null ? datasets[ds].size :
                Math.min(size, datasets[ds].size);
        }
    }
    return datasetFromIteratorFn(async () => {
        const streams = await deepMapAndAwaitAll(datasets, d => {
            if (d instanceof Dataset) {
                return { value: d.iterator(), recurse: false };
            }
            else if (isIterable(d)) {
                return { value: null, recurse: true };
            }
            else {
                throw new Error('Leaves of the structure passed to zip() must be Datasets, ' +
                    'not primitives.');
            }
        });
        return iteratorFromZipped(streams, ZipMismatchMode.SHORTEST);
    }, size);
}
/**
 * A zip function for use with deepZip, passed via the columnMajorBatch call.
 *
 * Accepts an array of identically-structured nested elements and either batches
 * them (if they are primitives, numeric arrays, or Tensors) or requests
 * recursion (if not).
 */
// tslint:disable-next-line:no-any
function deepBatchConcat(rows) {
    if (rows === null) {
        return null;
    }
    // use the first item to decide whether to recurse or batch here.
    const exampleRow = rows[0];
    if (canTensorify(exampleRow)) {
        // rows is an array of primitives, Tensors, or arrays.  Batch them.
        const value = batchConcat(rows);
        return { value, recurse: false };
    }
    // the example row is an object, so recurse into it.
    return { value: null, recurse: true };
}
/**
 * Assembles a list of same-shaped numbers, number arrays, or Tensors
 * into a single new Tensor where axis 0 is the batch dimension.
 */
function batchConcat(arrays) {
    if (arrays.length === 0) {
        // We can't return an empty Tensor because we don't know the element shape.
        throw new Error('Can\'t make a batch of zero elements.');
    }
    if (arrays[0] instanceof tf.Tensor) {
        // Input is an array of Tensors
        return tf.stack(arrays);
    }
    else {
        // Input is a possibly-nested array of numbers.
        return tf.tensor(arrays);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGF0YXNldC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtZGF0YS9zcmMvZGF0YXNldC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7OztHQWdCRztBQUVILE9BQU8sS0FBSyxFQUFFLE1BQU0sdUJBQXVCLENBQUM7QUFFNUMsT0FBTyxLQUFLLFVBQVUsTUFBTSxZQUFZLENBQUM7QUFFekMsT0FBTyxFQUFDLHdCQUF3QixFQUFFLG9CQUFvQixFQUFFLGlCQUFpQixFQUFFLGtCQUFrQixFQUFnQixlQUFlLEVBQUMsTUFBTSwyQkFBMkIsQ0FBQztBQUUvSixPQUFPLEVBQUMsWUFBWSxFQUFFLGtCQUFrQixFQUFpQixVQUFVLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQU81RixxRUFBcUU7QUFFckU7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTJCRztBQUNILE1BQU0sT0FBZ0IsT0FBTztJQUE3QjtRQVdXLFNBQUksR0FBVyxJQUFJLENBQUM7SUEyYy9CLENBQUM7SUF6Y0Msd0VBQXdFO0lBQ3hFLDJFQUEyRTtJQUMzRSw0RUFBNEU7SUFDNUUsMkNBQTJDO0lBQzNDLHVDQUF1QztJQUV2Qzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O09Bc0RHO0lBQ0gsS0FBSyxDQUFDLFNBQWlCLEVBQUUsY0FBYyxHQUFHLElBQUk7UUFDNUMsTUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDO1FBQ2xCLEVBQUUsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUNWLFNBQVMsR0FBRyxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUM7UUFDckIsU0FBUyxFQUFFLENBQUMsQ0FBQztRQUNqQixJQUFJLElBQUksQ0FBQztRQUNULElBQUksSUFBSSxDQUFDLElBQUksS0FBSyxRQUFRLElBQUksSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLEVBQUU7WUFDL0MsMEVBQTBFO1lBQzFFLFFBQVE7WUFDUixJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQztTQUNsQjthQUFNLElBQUksY0FBYyxFQUFFO1lBQ3pCLHlFQUF5RTtZQUN6RSxnREFBZ0Q7WUFDaEQsSUFBSSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksR0FBRyxTQUFTLENBQUMsQ0FBQztTQUN6QzthQUFNO1lBQ0wseUVBQXlFO1lBQ3pFLG9DQUFvQztZQUNwQyxJQUFJLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsSUFBSSxHQUFHLFNBQVMsQ0FBQyxDQUFDO1NBQzFDO1FBQ0QsT0FBTyxxQkFBcUIsQ0FBQyxLQUFLLElBQUksRUFBRTtZQUN0QyxPQUFPLENBQUMsTUFBTSxJQUFJLENBQUMsUUFBUSxFQUFFLENBQUM7aUJBQ3pCLGdCQUFnQixDQUFDLFNBQVMsRUFBRSxjQUFjLEVBQUUsZUFBZSxDQUFDLENBQUM7UUFDcEUsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ1gsQ0FBQztJQUVEOzs7Ozs7Ozs7Ozs7OztPQWNHO0lBQ0gsV0FBVyxDQUFDLE9BQW1CO1FBQzdCLE1BQU0sSUFBSSxHQUFHLElBQUksQ0FBQztRQUNsQixJQUFJLElBQUksQ0FBQztRQUNULElBQUksSUFBSSxDQUFDLElBQUksS0FBSyxRQUFRLElBQUksT0FBTyxDQUFDLElBQUksS0FBSyxRQUFRLEVBQUU7WUFDdkQsbUVBQW1FO1lBQ25FLFlBQVk7WUFDWixJQUFJLEdBQUcsUUFBUSxDQUFDO1NBQ2pCO2FBQU0sSUFBSSxJQUFJLENBQUMsSUFBSSxJQUFJLElBQUksSUFBSSxPQUFPLENBQUMsSUFBSSxJQUFJLElBQUksRUFBRTtZQUNwRCx1RUFBdUU7WUFDdkUsc0NBQXNDO1lBQ3RDLElBQUksR0FBRyxJQUFJLENBQUMsSUFBSSxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUM7U0FDakM7YUFBTTtZQUNMLDBFQUEwRTtZQUMxRSxnREFBZ0Q7WUFDaEQsSUFBSSxHQUFHLElBQUksQ0FBQztTQUNiO1FBQ0QsT0FBTyxxQkFBcUIsQ0FDeEIsS0FBSyxJQUFJLEVBQUUsQ0FDUCxDQUFDLE1BQU0sSUFBSSxDQUFDLFFBQVEsRUFBRSxDQUFDLENBQUMsV0FBVyxDQUFDLE1BQU0sT0FBTyxDQUFDLFFBQVEsRUFBRSxDQUFDLEVBQ2pFLElBQUksQ0FBQyxDQUFDO0lBQ1osQ0FBQztJQUVEOzs7Ozs7Ozs7Ozs7Ozs7T0FlRztJQUNILE1BQU0sQ0FBQyxTQUFnQztRQUNyQyxNQUFNLElBQUksR0FBRyxJQUFJLENBQUM7UUFDbEIsSUFBSSxJQUFJLENBQUM7UUFDVCxJQUFJLElBQUksQ0FBQyxJQUFJLEtBQUssUUFBUSxFQUFFO1lBQzFCLGdFQUFnRTtZQUNoRSxJQUFJLEdBQUcsUUFBUSxDQUFDO1NBQ2pCO2FBQU07WUFDTCwwRUFBMEU7WUFDMUUsc0JBQXNCO1lBQ3RCLElBQUksR0FBRyxJQUFJLENBQUM7U0FDYjtRQUNELE9BQU8scUJBQXFCLENBQUMsS0FBSyxJQUFJLEVBQUU7WUFDdEMsT0FBTyxDQUFDLE1BQU0sSUFBSSxDQUFDLFFBQVEsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzFFLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQztJQUNYLENBQUM7SUFFRDs7Ozs7Ozs7Ozs7Ozs7O09BZUc7SUFDSCxLQUFLLENBQUMsWUFBWSxDQUFDLENBQXFCO1FBQ3RDLE9BQU8sQ0FBQyxNQUFNLElBQUksQ0FBQyxRQUFRLEVBQUUsQ0FBQyxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNqRCxDQUFDO0lBRUQ7Ozs7Ozs7Ozs7Ozs7O09BY0c7SUFDSCxHQUFHLENBQStCLFNBQTBCO1FBQzFELE1BQU0sSUFBSSxHQUFHLElBQUksQ0FBQztRQUNsQixPQUFPLHFCQUFxQixDQUFDLEtBQUssSUFBSSxFQUFFO1lBQ3RDLE9BQU8sQ0FBQyxNQUFNLElBQUksQ0FBQyxRQUFRLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN2RSxDQUFDLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQ2hCLENBQUM7SUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztPQXNCRztJQUNILFFBQVEsQ0FBK0IsU0FBbUM7UUFFeEUsTUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDO1FBQ2xCLE9BQU8scUJBQXFCLENBQUMsS0FBSyxJQUFJLEVBQUU7WUFDdEMsT0FBTyxDQUFDLE1BQU0sSUFBSSxDQUFDLFFBQVEsRUFBRSxDQUFDLENBQUMsUUFBUSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQ3JELENBQUMsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDaEIsQ0FBQztJQUVEOzs7Ozs7OztPQVFHO0lBQ0gsUUFBUSxDQUFDLFVBQWtCO1FBQ3pCLElBQUksVUFBVSxJQUFJLElBQUksRUFBRTtZQUN0QixNQUFNLElBQUksVUFBVSxDQUNoQiwyREFBMkQsQ0FBQyxDQUFDO1NBQ2xFO1FBRUQsTUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDO1FBQ2xCLE9BQU8scUJBQXFCLENBQ3hCLEtBQUssSUFBSSxFQUFFLENBQUMsQ0FBQyxNQUFNLElBQUksQ0FBQyxRQUFRLEVBQUUsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxVQUFVLENBQUMsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDM0UsQ0FBQztJQUVEOzs7Ozs7Ozs7Ozs7Ozs7OztPQWlCRztJQUNILE1BQU0sQ0FBQyxLQUFjO1FBQ25CLE1BQU0sSUFBSSxHQUFHLElBQUksQ0FBQztRQUNsQixJQUFJLElBQUksQ0FBQztRQUNULElBQUksSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLElBQUksS0FBSyxHQUFHLENBQUMsRUFBRTtZQUNsQyxzRUFBc0U7WUFDdEUsc0VBQXNFO1lBQ3RFLFlBQVk7WUFDWixJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksR0FBRyxLQUFLLENBQUM7U0FDMUI7YUFBTSxJQUFJLEtBQUssS0FBSyxDQUFDLEVBQUU7WUFDdEIsZ0NBQWdDO1lBQ2hDLElBQUksR0FBRyxDQUFDLENBQUM7U0FDVjthQUFNLElBQUksSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLElBQUksQ0FBQyxLQUFLLEtBQUssU0FBUyxJQUFJLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRTtZQUNsRSxtRUFBbUU7WUFDbkUsa0VBQWtFO1lBQ2xFLElBQUksR0FBRyxRQUFRLENBQUM7U0FDakI7YUFBTTtZQUNMLHVFQUF1RTtZQUN2RSxJQUFJLEdBQUcsSUFBSSxDQUFDO1NBQ2I7UUFDRCxPQUFPLHFCQUFxQixDQUFDLEtBQUssSUFBSSxFQUFFO1lBQ3RDLE1BQU0sZ0JBQWdCLEdBQUcsb0JBQW9CLENBQ3pDLEtBQUssSUFBSSxFQUFFLENBQUMsQ0FBQyxFQUFDLEtBQUssRUFBRSxNQUFNLElBQUksQ0FBQyxRQUFRLEVBQUUsRUFBRSxJQUFJLEVBQUUsS0FBSyxFQUFDLENBQUMsQ0FBQyxDQUFDO1lBQy9ELE9BQU8sd0JBQXdCLENBQUMsZ0JBQWdCLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7UUFDaEUsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ1gsQ0FBQztJQUVEOzs7Ozs7Ozs7Ozs7Ozs7O09BZ0JHO0lBQ0gsSUFBSSxDQUFDLEtBQWE7UUFDaEIsTUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDO1FBQ2xCLElBQUksSUFBSSxDQUFDO1FBQ1QsSUFBSSxJQUFJLENBQUMsSUFBSSxJQUFJLElBQUksSUFBSSxLQUFLLElBQUksQ0FBQyxJQUFJLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxFQUFFO1lBQ3pELHVFQUF1RTtZQUN2RSx5RUFBeUU7WUFDekUsNEJBQTRCO1lBQzVCLElBQUksR0FBRyxJQUFJLENBQUMsSUFBSSxHQUFHLEtBQUssQ0FBQztTQUMxQjthQUFNLElBQ0gsSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJO1lBQ2pCLENBQUMsSUFBSSxDQUFDLElBQUksR0FBRyxLQUFLLElBQUksS0FBSyxLQUFLLFNBQVMsSUFBSSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEVBQUU7WUFDM0QsaUVBQWlFO1lBQ2pFLHlFQUF5RTtZQUN6RSxJQUFJLEdBQUcsQ0FBQyxDQUFDO1NBQ1Y7YUFBTTtZQUNMLHVFQUF1RTtZQUN2RSxJQUFJLEdBQUcsSUFBSSxDQUFDO1NBQ2I7UUFDRCxPQUFPLHFCQUFxQixDQUN4QixLQUFLLElBQUksRUFBRSxDQUFDLENBQUMsTUFBTSxJQUFJLENBQUMsUUFBUSxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDN0QsQ0FBQztJQU1EOzs7Ozs7Ozs7Ozs7Ozs7Ozs7OztPQW9CRztJQUNILE9BQU8sQ0FBQyxVQUFrQixFQUFFLElBQWEsRUFBRSxzQkFBc0IsR0FBRyxJQUFJO1FBRXRFLElBQUksVUFBVSxJQUFJLElBQUksSUFBSSxVQUFVLEdBQUcsQ0FBQyxFQUFFO1lBQ3hDLElBQUksSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLEVBQUU7Z0JBQ3JCLE1BQU0sSUFBSSxVQUFVLENBQ2hCLDBEQUEwRCxDQUFDLENBQUM7YUFDakU7aUJBQU07Z0JBQ0wsTUFBTSxJQUFJLFVBQVUsQ0FDaEIsNERBQTREO29CQUM1RCw2REFBNkQ7b0JBQzdELHlEQUF5RDtvQkFDekQsbUNBQW1DLElBQUksQ0FBQyxJQUFJLFlBQVksQ0FBQyxDQUFDO2FBQy9EO1NBQ0Y7UUFDRCxNQUFNLElBQUksR0FBRyxJQUFJLENBQUM7UUFDbEIsTUFBTSxNQUFNLEdBQUcsVUFBVSxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLEVBQUUsQ0FBQyxDQUFDO1FBQ2pFLE9BQU8scUJBQXFCLENBQUMsS0FBSyxJQUFJLEVBQUU7WUFDdEMsSUFBSSxLQUFLLEdBQUcsTUFBTSxDQUFDLEtBQUssRUFBRSxDQUFDO1lBQzNCLElBQUksc0JBQXNCLEVBQUU7Z0JBQzFCLEtBQUssSUFBSSxNQUFNLENBQUMsS0FBSyxFQUFFLENBQUM7YUFDekI7WUFDRCxPQUFPLENBQUMsTUFBTSxJQUFJLENBQUMsUUFBUSxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsVUFBVSxFQUFFLEtBQUssQ0FBQyxRQUFRLEVBQUUsQ0FBQyxDQUFDO1FBQ3ZFLENBQUMsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDaEIsQ0FBQztJQUVEOzs7Ozs7Ozs7Ozs7Ozs7O09BZ0JHO0lBQ0gsSUFBSSxDQUFDLEtBQWE7UUFDaEIsTUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDO1FBQ2xCLElBQUksSUFBSSxDQUFDO1FBQ1QsSUFBSSxJQUFJLENBQUMsSUFBSSxJQUFJLElBQUksSUFBSSxJQUFJLENBQUMsSUFBSSxHQUFHLEtBQUssRUFBRTtZQUMxQyx1RUFBdUU7WUFDdkUsaUJBQWlCO1lBQ2pCLElBQUksR0FBRyxLQUFLLENBQUM7U0FDZDthQUFNLElBQUksSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLElBQUksSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLEVBQUU7WUFDbEQsc0VBQXNFO1lBQ3RFLDhDQUE4QztZQUM5QyxJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQztTQUNsQjthQUFNO1lBQ0wsdUVBQXVFO1lBQ3ZFLElBQUksR0FBRyxJQUFJLENBQUM7U0FDYjtRQUNELE9BQU8scUJBQXFCLENBQ3hCLEtBQUssSUFBSSxFQUFFLENBQUMsQ0FBQyxNQUFNLElBQUksQ0FBQyxRQUFRLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQztJQUM3RCxDQUFDO0lBRUQ7Ozs7Ozs7Ozs7Ozs7OztPQWVHO0lBQ0gsS0FBSyxDQUFDLE9BQU87UUFDWCxJQUFJLElBQUksQ0FBQyxJQUFJLEtBQUssUUFBUSxFQUFFO1lBQzFCLE1BQU0sSUFBSSxLQUFLLENBQUMsZ0RBQWdELENBQUMsQ0FBQztTQUNuRTtRQUNELE9BQU8sQ0FBQyxNQUFNLElBQUksQ0FBQyxRQUFRLEVBQUUsQ0FBQyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQzNDLENBQUM7SUFFRDs7Ozs7Ozs7OztPQVVHO0lBQ0gsS0FBSyxDQUFDLGNBQWM7UUFDbEIsSUFBSSxJQUFJLENBQUMsSUFBSSxLQUFLLFFBQVEsRUFBRTtZQUMxQixNQUFNLElBQUksS0FBSyxDQUFDLGdEQUFnRCxDQUFDLENBQUM7U0FDbkU7UUFDRCxPQUFPLENBQUMsTUFBTSxJQUFJLENBQUMsUUFBUSxFQUFFLENBQUMsQ0FBQyxjQUFjLEVBQUUsQ0FBQztJQUNsRCxDQUFDOztBQTdIRCx1REFBdUQ7QUFFdkMsdUJBQWUsR0FBRyxLQUFLLENBQUM7QUE4SDFDOzs7Ozs7Ozs7OztHQVdHO0FBQ0gsTUFBTSxVQUFVLHFCQUFxQixDQUNqQyxVQUEwQyxFQUMxQyxPQUFlLElBQUk7SUFDckIsT0FBTyxJQUFJLEtBQU0sU0FBUSxPQUFVO1FBQXhCOztZQUNBLFNBQUksR0FBRyxJQUFJLENBQUM7UUFTdkIsQ0FBQztRQVBDOzs7V0FHRztRQUNILEtBQUssQ0FBQyxRQUFRO1lBQ1osT0FBTyxVQUFVLEVBQUUsQ0FBQztRQUN0QixDQUFDO0tBQ0YsRUFDQyxDQUFDO0FBQ0wsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7OztHQWlCRztBQUNILE1BQU0sVUFBVSxLQUFLLENBQStCLEtBQVU7SUFDNUQsT0FBTyxxQkFBcUIsQ0FDeEIsS0FBSyxJQUFJLEVBQUUsQ0FBQyxpQkFBaUIsQ0FBQyxLQUFLLENBQUMsRUFBRSxLQUFLLENBQUMsTUFBTSxDQUFDLENBQUM7QUFDMUQsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBd0NHO0FBQ0gsTUFBTSxVQUFVLEdBQUcsQ0FBK0IsUUFBMEI7SUFFMUUsZ0RBQWdEO0lBQ2hELElBQUksQ0FBQyxVQUFVLENBQUMsUUFBUSxDQUFDLEVBQUU7UUFDekIsTUFBTSxJQUFJLEtBQUssQ0FBQyxtREFBbUQsQ0FBQyxDQUFDO0tBQ3RFO0lBQ0QsSUFBSSxJQUFJLENBQUM7SUFDVCxJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsUUFBUSxDQUFDLEVBQUU7UUFDM0IsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUU7WUFDeEMsSUFBSSxHQUFHLElBQUksSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFFLFFBQVEsQ0FBQyxDQUFDLENBQWdCLENBQUMsSUFBSSxDQUFDLENBQUM7Z0JBQ2xDLElBQUksQ0FBQyxHQUFHLENBQUMsSUFBSSxFQUFHLFFBQVEsQ0FBQyxDQUFDLENBQWdCLENBQUMsSUFBSSxDQUFDLENBQUM7U0FDeEU7S0FDRjtTQUFNLElBQUksUUFBUSxZQUFZLE1BQU0sRUFBRTtRQUNyQyxLQUFLLE1BQU0sRUFBRSxJQUFJLFFBQVEsRUFBRTtZQUN6QixJQUFJLEdBQUcsSUFBSSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUUsUUFBUSxDQUFDLEVBQUUsQ0FBZ0IsQ0FBQyxJQUFJLENBQUMsQ0FBQztnQkFDbkMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxJQUFJLEVBQUcsUUFBUSxDQUFDLEVBQUUsQ0FBZ0IsQ0FBQyxJQUFJLENBQUMsQ0FBQztTQUN6RTtLQUNGO0lBQ0QsT0FBTyxxQkFBcUIsQ0FBSSxLQUFLLElBQUksRUFBRTtRQUN6QyxNQUFNLE9BQU8sR0FBRyxNQUFNLGtCQUFrQixDQUFDLFFBQVEsRUFBRSxDQUFDLENBQUMsRUFBRTtZQUNyRCxJQUFJLENBQUMsWUFBWSxPQUFPLEVBQUU7Z0JBQ3hCLE9BQU8sRUFBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLFFBQVEsRUFBRSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsQ0FBQzthQUM5QztpQkFBTSxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUMsRUFBRTtnQkFDeEIsT0FBTyxFQUFDLEtBQUssRUFBRSxJQUFJLEVBQUUsT0FBTyxFQUFFLElBQUksRUFBQyxDQUFDO2FBQ3JDO2lCQUFNO2dCQUNMLE1BQU0sSUFBSSxLQUFLLENBQ1gsNERBQTREO29CQUM1RCxpQkFBaUIsQ0FBQyxDQUFDO2FBQ3hCO1FBQ0gsQ0FBQyxDQUFDLENBQUM7UUFDSCxPQUFPLGtCQUFrQixDQUFJLE9BQU8sRUFBRSxlQUFlLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDbEUsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDO0FBQ1gsQ0FBQztBQUVEOzs7Ozs7R0FNRztBQUNILGtDQUFrQztBQUNsQyxTQUFTLGVBQWUsQ0FBQyxJQUFXO0lBQ2xDLElBQUksSUFBSSxLQUFLLElBQUksRUFBRTtRQUNqQixPQUFPLElBQUksQ0FBQztLQUNiO0lBRUQsaUVBQWlFO0lBQ2pFLE1BQU0sVUFBVSxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUUzQixJQUFJLFlBQVksQ0FBQyxVQUFVLENBQUMsRUFBRTtRQUM1QixtRUFBbUU7UUFDbkUsTUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ2hDLE9BQU8sRUFBQyxLQUFLLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxDQUFDO0tBQ2hDO0lBRUQsb0RBQW9EO0lBQ3BELE9BQU8sRUFBQyxLQUFLLEVBQUUsSUFBSSxFQUFFLE9BQU8sRUFBRSxJQUFJLEVBQUMsQ0FBQztBQUN0QyxDQUFDO0FBRUQ7OztHQUdHO0FBQ0gsU0FBUyxXQUFXLENBQW9DLE1BQVc7SUFFakUsSUFBSSxNQUFNLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtRQUN2QiwyRUFBMkU7UUFDM0UsTUFBTSxJQUFJLEtBQUssQ0FBQyx1Q0FBdUMsQ0FBQyxDQUFDO0tBQzFEO0lBRUQsSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDLFlBQVksRUFBRSxDQUFDLE1BQU0sRUFBRTtRQUNsQywrQkFBK0I7UUFDL0IsT0FBTyxFQUFFLENBQUMsS0FBSyxDQUFDLE1BQXFCLENBQUMsQ0FBQztLQUN4QztTQUFNO1FBQ0wsK0NBQStDO1FBQy9DLE9BQU8sRUFBRSxDQUFDLE1BQU0sQ0FBQyxNQUFvQixDQUFDLENBQUM7S0FDeEM7QUFDSCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICpcbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICogYXMgdGYgZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7VGVuc29yQ29udGFpbmVyLCBUZW5zb3JMaWtlfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0ICogYXMgc2VlZHJhbmRvbSBmcm9tICdzZWVkcmFuZG9tJztcblxuaW1wb3J0IHtpdGVyYXRvckZyb21Db25jYXRlbmF0ZWQsIGl0ZXJhdG9yRnJvbUZ1bmN0aW9uLCBpdGVyYXRvckZyb21JdGVtcywgaXRlcmF0b3JGcm9tWmlwcGVkLCBMYXp5SXRlcmF0b3IsIFppcE1pc21hdGNoTW9kZX0gZnJvbSAnLi9pdGVyYXRvcnMvbGF6eV9pdGVyYXRvcic7XG5pbXBvcnQge0NvbnRhaW5lcn0gZnJvbSAnLi90eXBlcyc7XG5pbXBvcnQge2NhblRlbnNvcmlmeSwgZGVlcE1hcEFuZEF3YWl0QWxsLCBEZWVwTWFwUmVzdWx0LCBpc0l0ZXJhYmxlfSBmcm9tICcuL3V0aWwvZGVlcF9tYXAnO1xuXG4vKipcbiAqIEEgbmVzdGVkIHN0cnVjdHVyZSBvZiBEYXRhc2V0cywgdXNlZCBhcyB0aGUgaW5wdXQgdG8gemlwKCkuXG4gKi9cbmV4cG9ydCB0eXBlIERhdGFzZXRDb250YWluZXIgPSBDb250YWluZXI8RGF0YXNldDxUZW5zb3JDb250YWluZXI+PjtcblxuLy8gVE9ETyhzb2VyZ2VsKTogY29uc2lkZXIgdmVjdG9yaXplZCBvcGVyYXRpb25zIHdpdGhpbiB0aGUgcGlwZWxpbmUuXG5cbi8qKlxuICogUmVwcmVzZW50cyBhIHBvdGVudGlhbGx5IGxhcmdlIGxpc3Qgb2YgaW5kZXBlbmRlbnQgZGF0YSBlbGVtZW50cyAodHlwaWNhbGx5XG4gKiAnc2FtcGxlcycgb3IgJ2V4YW1wbGVzJykuXG4gKlxuICogQSAnZGF0YSBleGFtcGxlJyBtYXkgYmUgYSBwcmltaXRpdmUsIGFuIGFycmF5LCBhIG1hcCBmcm9tIHN0cmluZyBrZXlzIHRvXG4gKiB2YWx1ZXMsIG9yIGFueSBuZXN0ZWQgc3RydWN0dXJlIG9mIHRoZXNlLlxuICpcbiAqIEEgYERhdGFzZXRgIHJlcHJlc2VudHMgYW4gb3JkZXJlZCBjb2xsZWN0aW9uIG9mIGVsZW1lbnRzLCB0b2dldGhlciB3aXRoIGFcbiAqIGNoYWluIG9mIHRyYW5zZm9ybWF0aW9ucyB0byBiZSBwZXJmb3JtZWQgb24gdGhvc2UgZWxlbWVudHMuIEVhY2hcbiAqIHRyYW5zZm9ybWF0aW9uIGlzIGEgbWV0aG9kIG9mIGBEYXRhc2V0YCB0aGF0IHJldHVybnMgYW5vdGhlciBgRGF0YXNldGAsIHNvXG4gKiB0aGVzZSBtYXkgYmUgY2hhaW5lZCwgZS5nLlxuICogYGNvbnN0IHByb2Nlc3NlZERhdGFzZXQgPSByYXdEYXRhc2V0LmZpbHRlciguLi4pLm1hcCguLi4pLmJhdGNoKC4uLilgLlxuICpcbiAqIERhdGEgbG9hZGluZyBhbmQgdHJhbnNmb3JtYXRpb24gaXMgZG9uZSBpbiBhIGxhenksIHN0cmVhbWluZyBmYXNoaW9uLiAgVGhlXG4gKiBkYXRhc2V0IG1heSBiZSBpdGVyYXRlZCBvdmVyIG11bHRpcGxlIHRpbWVzOyBlYWNoIGl0ZXJhdGlvbiBzdGFydHMgdGhlIGRhdGFcbiAqIGxvYWRpbmcgYW5ldyBhbmQgcmVjYXBpdHVsYXRlcyB0aGUgdHJhbnNmb3JtYXRpb25zLlxuICpcbiAqIEEgYERhdGFzZXRgIGlzIHR5cGljYWxseSBwcm9jZXNzZWQgYXMgYSBzdHJlYW0gb2YgdW5iYXRjaGVkIGV4YW1wbGVzIC0tIGkuZS4sXG4gKiBpdHMgdHJhbnNmb3JtYXRpb25zIGFyZSBhcHBsaWVkIG9uZSBleGFtcGxlIGF0IGEgdGltZS4gQmF0Y2hpbmcgcHJvZHVjZXMgYVxuICogbmV3IGBEYXRhc2V0YCB3aGVyZSBlYWNoIGVsZW1lbnQgaXMgYSBiYXRjaC4gQmF0Y2hpbmcgc2hvdWxkIHVzdWFsbHkgY29tZVxuICogbGFzdCBpbiBhIHBpcGVsaW5lLCBiZWNhdXNlIGRhdGEgdHJhbnNmb3JtYXRpb25zIGFyZSBlYXNpZXIgdG8gZXhwcmVzcyBvbiBhXG4gKiBwZXItZXhhbXBsZSBiYXNpcyB0aGFuIG9uIGEgcGVyLWJhdGNoIGJhc2lzLlxuICpcbiAqIFRoZSBmb2xsb3dpbmcgY29kZSBleGFtcGxlcyBhcmUgY2FsbGluZyBgYXdhaXQgZGF0YXNldC5mb3JFYWNoQXN5bmMoLi4uKWAgdG9cbiAqIGl0ZXJhdGUgb25jZSBvdmVyIHRoZSBlbnRpcmUgZGF0YXNldCBpbiBvcmRlciB0byBwcmludCBvdXQgdGhlIGRhdGEuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0RhdGEnLCBzdWJoZWFkaW5nOiAnQ2xhc3NlcycsIG5hbWVzcGFjZTogJ2RhdGEnfVxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgRGF0YXNldDxUIGV4dGVuZHMgdGYuVGVuc29yQ29udGFpbmVyPiB7XG4gIC8qXG4gICAqIFByb3ZpZGUgYSBuZXcgc3RyZWFtIG9mIGVsZW1lbnRzLiAgTm90ZSB0aGlzIHdpbGwgYWxzbyBzdGFydCBuZXcgc3RyZWFtc1xuICAgKiBmcm9tIGFueSB1bmRlcmx5aW5nIGBEYXRhc2V0YHMuXG4gICAqXG4gICAqIENBVVRJT046IEFueSBUZW5zb3JzIGNvbnRhaW5lZCB3aXRoaW4gdGhlIGVsZW1lbnRzIHJldHVybmVkIGZyb21cbiAgICogdGhpcyBzdHJlYW0gKm11c3QqIGJlIG1hbnVhbGx5IGRpc3Bvc2VkIHRvIGF2b2lkIGEgR1BVIG1lbW9yeSBsZWFrLlxuICAgKiBUaGUgdGYudGlkeSgpIGFwcHJvYWNoIGNhbm5vdCBiZSB1c2VkIGluIGFuIGFzeW5jaHJvbm91cyBjb250ZXh0LlxuICAgKi9cbiAgYWJzdHJhY3QgaXRlcmF0b3IoKTogUHJvbWlzZTxMYXp5SXRlcmF0b3I8VD4+O1xuXG4gIHJlYWRvbmx5IHNpemU6IG51bWJlciA9IG51bGw7XG5cbiAgLy8gVE9ETyhzb2VyZ2VsKTogTWFrZSBEYXRhc2V0cyByZXBvcnQgd2hldGhlciByZXBlYXRlZCBpdGVyYXRvcigpIGNhbGxzXG4gIC8vIHByb2R1Y2UgdGhlIHNhbWUgcmVzdWx0IChlLmcuLCByZWFkaW5nIGZyb20gYSBmaWxlKSBvciBkaWZmZXJlbnQgcmVzdWx0c1xuICAvLyAoZS5nLiwgZnJvbSB0aGUgd2ViY2FtKS4gIEN1cnJlbnRseSB3ZSBkb24ndCBtYWtlIHRoaXMgZGlzdGluY3Rpb24gYnV0IGl0XG4gIC8vIGNvdWxkIGJlIGltcG9ydGFudCBmb3IgdGhlIHVzZXIgdG8ga25vdy5cbiAgLy8gYWJzdHJhY3QgaXNEZXRlcm1pbmlzdGljKCk6IGJvb2xlYW47XG5cbiAgLyoqXG4gICAqIEdyb3VwcyBlbGVtZW50cyBpbnRvIGJhdGNoZXMuXG4gICAqXG4gICAqIEl0IGlzIGFzc3VtZWQgdGhhdCBlYWNoIG9mIHRoZSBpbmNvbWluZyBkYXRhc2V0IGVsZW1lbnRzIGhhcyB0aGUgc2FtZVxuICAgKiBzdHJ1Y3R1cmUgLS0gaS5lLiB0aGUgc2FtZSBzZXQgb2Yga2V5cyBhdCBlYWNoIGxvY2F0aW9uIGluIGFuIG9iamVjdFxuICAgKiBoaWVyYXJjaHkuICBGb3IgZWFjaCBrZXksIHRoZSByZXN1bHRpbmcgYERhdGFzZXRgIHByb3ZpZGVzIGEgYmF0Y2hlZFxuICAgKiBlbGVtZW50IGNvbGxlY3RpbmcgYWxsIG9mIHRoZSBpbmNvbWluZyB2YWx1ZXMgZm9yIHRoYXQga2V5LlxuICAgKlxuICAgKiAgKiBJbmNvbWluZyBwcmltaXRpdmVzIGFyZSBncm91cGVkIGludG8gYSAxLUQgVGVuc29yLlxuICAgKiAgKiBJbmNvbWluZyBUZW5zb3JzIGFyZSBncm91cGVkIGludG8gYSBuZXcgVGVuc29yIHdoZXJlIHRoZSAwdGggYXhpcyBpc1xuICAgKiAgICB0aGUgYmF0Y2ggZGltZW5zaW9uLlxuICAgKiAgKiBJbmNvbWluZyBhcnJheXMgYXJlIGNvbnZlcnRlZCB0byBUZW5zb3IgYW5kIHRoZW4gYmF0Y2hlZC5cbiAgICogICogQSBuZXN0ZWQgYXJyYXkgaXMgaW50ZXJwcmV0ZWQgYXMgYW4gbi1EIFRlbnNvciwgc28gdGhlIGJhdGNoZWQgcmVzdWx0XG4gICAqICAgIGhhcyBuKzEgZGltZW5zaW9ucy5cbiAgICogICogQW4gYXJyYXkgdGhhdCBjYW5ub3QgYmUgY29udmVydGVkIHRvIFRlbnNvciBwcm9kdWNlcyBhbiBlcnJvci5cbiAgICpcbiAgICogSWYgYW4gYXJyYXkgc2hvdWxkIG5vdCBiZSBiYXRjaGVkIGFzIGEgdW5pdCwgaXQgc2hvdWxkIGZpcnN0IGJlIGNvbnZlcnRlZFxuICAgKiB0byBhbiBvYmplY3Qgd2l0aCBpbnRlZ2VyIGtleXMuXG4gICAqXG4gICAqIEhlcmUgYXJlIGEgZmV3IGV4YW1wbGVzOlxuICAgKlxuICAgKiBCYXRjaCBhIGRhdGFzZXQgb2YgbnVtYmVyczpcbiAgICogYGBganNcbiAgICogY29uc3QgYSA9IHRmLmRhdGEuYXJyYXkoWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDhdKS5iYXRjaCg0KTtcbiAgICogYXdhaXQgYS5mb3JFYWNoQXN5bmMoZSA9PiBlLnByaW50KCkpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQmF0Y2ggYSBkYXRhc2V0IG9mIGFycmF5czpcbiAgICogYGBganNcbiAgICogY29uc3QgYiA9IHRmLmRhdGEuYXJyYXkoW1sxXSwgWzJdLCBbM10sIFs0XSwgWzVdLCBbNl0sIFs3XSwgWzhdXSkuYmF0Y2goNCk7XG4gICAqIGF3YWl0IGIuZm9yRWFjaEFzeW5jKGUgPT4gZS5wcmludCgpKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEJhdGNoIGEgZGF0YXNldCBvZiBvYmplY3RzOlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBjID0gdGYuZGF0YS5hcnJheShbe2E6IDEsIGI6IDExfSwge2E6IDIsIGI6IDEyfSwge2E6IDMsIGI6IDEzfSxcbiAgICogICB7YTogNCwgYjogMTR9LCB7YTogNSwgYjogMTV9LCB7YTogNiwgYjogMTZ9LCB7YTogNywgYjogMTd9LFxuICAgKiAgIHthOiA4LCBiOiAxOH1dKS5iYXRjaCg0KTtcbiAgICogYXdhaXQgYy5mb3JFYWNoQXN5bmMoZSA9PiB7XG4gICAqICAgY29uc29sZS5sb2coJ3snKTtcbiAgICogICBmb3IodmFyIGtleSBpbiBlKSB7XG4gICAqICAgICBjb25zb2xlLmxvZyhrZXkrJzonKTtcbiAgICogICAgIGVba2V5XS5wcmludCgpO1xuICAgKiAgIH1cbiAgICogICBjb25zb2xlLmxvZygnfScpO1xuICAgKiB9KVxuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIGJhdGNoU2l6ZSBUaGUgbnVtYmVyIG9mIGVsZW1lbnRzIGRlc2lyZWQgcGVyIGJhdGNoLlxuICAgKiBAcGFyYW0gc21hbGxMYXN0QmF0Y2ggV2hldGhlciB0byBlbWl0IHRoZSBmaW5hbCBiYXRjaCB3aGVuIGl0IGhhcyBmZXdlclxuICAgKiAgIHRoYW4gYmF0Y2hTaXplIGVsZW1lbnRzLiBEZWZhdWx0IHRydWUuXG4gICAqIEByZXR1cm5zIEEgYERhdGFzZXRgLCBmcm9tIHdoaWNoIGEgc3RyZWFtIG9mIGJhdGNoZXMgY2FuIGJlIG9idGFpbmVkLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGJhdGNoKGJhdGNoU2l6ZTogbnVtYmVyLCBzbWFsbExhc3RCYXRjaCA9IHRydWUpOiBEYXRhc2V0PHRmLlRlbnNvckNvbnRhaW5lcj4ge1xuICAgIGNvbnN0IGJhc2UgPSB0aGlzO1xuICAgIHRmLnV0aWwuYXNzZXJ0KFxuICAgICAgICBiYXRjaFNpemUgPiAwLCAoKSA9PiBgYmF0Y2hTaXplIG5lZWRzIHRvIGJlIHBvc2l0aXZlLCBidXQgaXQgaXNcbiAgICAgICR7YmF0Y2hTaXplfWApO1xuICAgIGxldCBzaXplO1xuICAgIGlmICh0aGlzLnNpemUgPT09IEluZmluaXR5IHx8IHRoaXMuc2l6ZSA9PSBudWxsKSB7XG4gICAgICAvLyBJZiB0aGUgc2l6ZSBvZiB0aGlzIGRhdGFzZXQgaXMgaW5maW5pdHkgb3IgbnVsbCwgdGhlIG5ldyBzaXplIGtlZXBzIHRoZVxuICAgICAgLy8gc2FtZS5cbiAgICAgIHNpemUgPSB0aGlzLnNpemU7XG4gICAgfSBlbHNlIGlmIChzbWFsbExhc3RCYXRjaCkge1xuICAgICAgLy8gSWYgdGhlIHNpemUgb2YgdGhpcyBkYXRhc2V0IGlzIGtub3duIGFuZCBpbmNsdWRlIHNtYWxsIGxhc3QgYmF0Y2gsIHRoZVxuICAgICAgLy8gbmV3IHNpemUgaXMgZnVsbCBiYXRjaCBjb3VudCBwbHVzIGxhc3QgYmF0Y2guXG4gICAgICBzaXplID0gTWF0aC5jZWlsKHRoaXMuc2l6ZSAvIGJhdGNoU2l6ZSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIElmIHRoZSBzaXplIG9mIHRoaXMgZGF0YXNldCBpcyBrbm93biBhbmQgbm90IGluY2x1ZGUgc21hbGwgbGFzdCBiYXRjaCxcbiAgICAgIC8vIHRoZSBuZXcgc2l6ZSBpcyBmdWxsIGJhdGNoIGNvdW50LlxuICAgICAgc2l6ZSA9IE1hdGguZmxvb3IodGhpcy5zaXplIC8gYmF0Y2hTaXplKTtcbiAgICB9XG4gICAgcmV0dXJuIGRhdGFzZXRGcm9tSXRlcmF0b3JGbihhc3luYyAoKSA9PiB7XG4gICAgICByZXR1cm4gKGF3YWl0IGJhc2UuaXRlcmF0b3IoKSlcbiAgICAgICAgICAuY29sdW1uTWFqb3JCYXRjaChiYXRjaFNpemUsIHNtYWxsTGFzdEJhdGNoLCBkZWVwQmF0Y2hDb25jYXQpO1xuICAgIH0sIHNpemUpO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbmNhdGVuYXRlcyB0aGlzIGBEYXRhc2V0YCB3aXRoIGFub3RoZXIuXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IGEgPSB0Zi5kYXRhLmFycmF5KFsxLCAyLCAzXSk7XG4gICAqIGNvbnN0IGIgPSB0Zi5kYXRhLmFycmF5KFs0LCA1LCA2XSk7XG4gICAqIGNvbnN0IGMgPSBhLmNvbmNhdGVuYXRlKGIpO1xuICAgKiBhd2FpdCBjLmZvckVhY2hBc3luYyhlID0+IGNvbnNvbGUubG9nKGUpKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSBkYXRhc2V0IEEgYERhdGFzZXRgIHRvIGJlIGNvbmNhdGVuYXRlZCBvbnRvIHRoaXMgb25lLlxuICAgKiBAcmV0dXJucyBBIGBEYXRhc2V0YC5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ0RhdGEnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBjb25jYXRlbmF0ZShkYXRhc2V0OiBEYXRhc2V0PFQ+KTogRGF0YXNldDxUPiB7XG4gICAgY29uc3QgYmFzZSA9IHRoaXM7XG4gICAgbGV0IHNpemU7XG4gICAgaWYgKHRoaXMuc2l6ZSA9PT0gSW5maW5pdHkgfHwgZGF0YXNldC5zaXplID09PSBJbmZpbml0eSkge1xuICAgICAgLy8gSWYgdGhlIHNpemUgb2YgYW55IG9mIHRoZXNlIHR3byBkYXRhc2V0IGlzIGluZmluaXR5LCBuZXcgc2l6ZSBpc1xuICAgICAgLy8gaW5maW5pdHkuXG4gICAgICBzaXplID0gSW5maW5pdHk7XG4gICAgfSBlbHNlIGlmICh0aGlzLnNpemUgIT0gbnVsbCAmJiBkYXRhc2V0LnNpemUgIT0gbnVsbCkge1xuICAgICAgLy8gSWYgdGhlIHNpemUgb2YgYm90aCBkYXRhc2V0cyBhcmUga25vd24gYW5kIG5vdCBpbmZpbml0eSwgbmV3IHNpemUgaXNcbiAgICAgIC8vIHN1bSB0aGUgc2l6ZSBvZiB0aGVzZSB0d28gZGF0YXNldHMuXG4gICAgICBzaXplID0gdGhpcy5zaXplICsgZGF0YXNldC5zaXplO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBJZiBuZWl0aGVyIG9mIHRoZXNlIHR3byBkYXRhc2V0cyBoYXMgaW5maW5pdGUgc2l6ZSBhbmQgYW55IG9mIHRoZXNlIHR3b1xuICAgICAgLy8gZGF0YXNldHMnIHNpemUgaXMgbnVsbCwgdGhlIG5ldyBzaXplIGlzIG51bGwuXG4gICAgICBzaXplID0gbnVsbDtcbiAgICB9XG4gICAgcmV0dXJuIGRhdGFzZXRGcm9tSXRlcmF0b3JGbihcbiAgICAgICAgYXN5bmMgKCkgPT5cbiAgICAgICAgICAgIChhd2FpdCBiYXNlLml0ZXJhdG9yKCkpLmNvbmNhdGVuYXRlKGF3YWl0IGRhdGFzZXQuaXRlcmF0b3IoKSksXG4gICAgICAgIHNpemUpO1xuICB9XG5cbiAgLyoqXG4gICAqIEZpbHRlcnMgdGhpcyBkYXRhc2V0IGFjY29yZGluZyB0byBgcHJlZGljYXRlYC5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgYSA9IHRmLmRhdGEuYXJyYXkoWzEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEwXSlcbiAgICogICAuZmlsdGVyKHggPT4geCUyID09PSAwKTtcbiAgICogYXdhaXQgYS5mb3JFYWNoQXN5bmMoZSA9PiBjb25zb2xlLmxvZyhlKSk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0gcHJlZGljYXRlIEEgZnVuY3Rpb24gbWFwcGluZyBhIGRhdGFzZXQgZWxlbWVudCB0byBhIGJvb2xlYW4gb3IgYVxuICAgKiBgUHJvbWlzZWAgZm9yIG9uZS5cbiAgICpcbiAgICogQHJldHVybnMgQSBgRGF0YXNldGAgb2YgZWxlbWVudHMgZm9yIHdoaWNoIHRoZSBwcmVkaWNhdGUgd2FzIHRydWUuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdEYXRhJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgZmlsdGVyKHByZWRpY2F0ZTogKHZhbHVlOiBUKSA9PiBib29sZWFuKTogRGF0YXNldDxUPiB7XG4gICAgY29uc3QgYmFzZSA9IHRoaXM7XG4gICAgbGV0IHNpemU7XG4gICAgaWYgKHRoaXMuc2l6ZSA9PT0gSW5maW5pdHkpIHtcbiAgICAgIC8vIElmIHRoZSBzaXplIG9mIHRoaXMgZGF0YXNldCBpcyBpbmZpbml0eSwgbmV3IHNpemUgaXMgaW5maW5pdHlcbiAgICAgIHNpemUgPSBJbmZpbml0eTtcbiAgICB9IGVsc2Uge1xuICAgICAgLy8gSWYgdGhpcyBkYXRhc2V0IGhhcyBsaW1pdGVkIGVsZW1lbnRzLCBuZXcgc2l6ZSBpcyBudWxsIGJlY2F1c2UgaXQgbWlnaHRcbiAgICAgIC8vIGV4aGF1c3RlZCByYW5kb21seS5cbiAgICAgIHNpemUgPSBudWxsO1xuICAgIH1cbiAgICByZXR1cm4gZGF0YXNldEZyb21JdGVyYXRvckZuKGFzeW5jICgpID0+IHtcbiAgICAgIHJldHVybiAoYXdhaXQgYmFzZS5pdGVyYXRvcigpKS5maWx0ZXIoeCA9PiB0Zi50aWR5KCgpID0+IHByZWRpY2F0ZSh4KSkpO1xuICAgIH0sIHNpemUpO1xuICB9XG5cbiAgLyoqXG4gICAqIEFwcGx5IGEgZnVuY3Rpb24gdG8gZXZlcnkgZWxlbWVudCBvZiB0aGUgZGF0YXNldC5cbiAgICpcbiAgICogQWZ0ZXIgdGhlIGZ1bmN0aW9uIGlzIGFwcGxpZWQgdG8gYSBkYXRhc2V0IGVsZW1lbnQsIGFueSBUZW5zb3JzIGNvbnRhaW5lZFxuICAgKiB3aXRoaW4gdGhhdCBlbGVtZW50IGFyZSBkaXNwb3NlZC5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgYSA9IHRmLmRhdGEuYXJyYXkoWzEsIDIsIDNdKTtcbiAgICogYXdhaXQgYS5mb3JFYWNoQXN5bmMoZSA9PiBjb25zb2xlLmxvZyhlKSk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0gZiBBIGZ1bmN0aW9uIHRvIGFwcGx5IHRvIGVhY2ggZGF0YXNldCBlbGVtZW50LlxuICAgKiBAcmV0dXJucyBBIGBQcm9taXNlYCB0aGF0IHJlc29sdmVzIGFmdGVyIGFsbCBlbGVtZW50cyBoYXZlIGJlZW4gcHJvY2Vzc2VkLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGFzeW5jIGZvckVhY2hBc3luYyhmOiAoaW5wdXQ6IFQpID0+IHZvaWQpOiBQcm9taXNlPHZvaWQ+IHtcbiAgICByZXR1cm4gKGF3YWl0IHRoaXMuaXRlcmF0b3IoKSkuZm9yRWFjaEFzeW5jKGYpO1xuICB9XG5cbiAgLyoqXG4gICAqIE1hcHMgdGhpcyBkYXRhc2V0IHRocm91Z2ggYSAxLXRvLTEgdHJhbnNmb3JtLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBhID0gdGYuZGF0YS5hcnJheShbMSwgMiwgM10pLm1hcCh4ID0+IHgqeCk7XG4gICAqIGF3YWl0IGEuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coZSkpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIHRyYW5zZm9ybSBBIGZ1bmN0aW9uIG1hcHBpbmcgYSBkYXRhc2V0IGVsZW1lbnQgdG8gYSB0cmFuc2Zvcm1lZFxuICAgKiAgIGRhdGFzZXQgZWxlbWVudC5cbiAgICpcbiAgICogQHJldHVybnMgQSBgRGF0YXNldGAgb2YgdHJhbnNmb3JtZWQgZWxlbWVudHMuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdEYXRhJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgbWFwPE8gZXh0ZW5kcyB0Zi5UZW5zb3JDb250YWluZXI+KHRyYW5zZm9ybTogKHZhbHVlOiBUKSA9PiBPKTogRGF0YXNldDxPPiB7XG4gICAgY29uc3QgYmFzZSA9IHRoaXM7XG4gICAgcmV0dXJuIGRhdGFzZXRGcm9tSXRlcmF0b3JGbihhc3luYyAoKSA9PiB7XG4gICAgICByZXR1cm4gKGF3YWl0IGJhc2UuaXRlcmF0b3IoKSkubWFwKHggPT4gdGYudGlkeSgoKSA9PiB0cmFuc2Zvcm0oeCkpKTtcbiAgICB9LCB0aGlzLnNpemUpO1xuICB9XG5cbiAgLyoqXG4gICAqIE1hcHMgdGhpcyBkYXRhc2V0IHRocm91Z2ggYW4gYXN5bmMgMS10by0xIHRyYW5zZm9ybS5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgYSA9XG4gICAqICB0Zi5kYXRhLmFycmF5KFsxLCAyLCAzXSkubWFwQXN5bmMoeCA9PiBuZXcgUHJvbWlzZShmdW5jdGlvbihyZXNvbHZlKXtcbiAgICogICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAqICAgICAgcmVzb2x2ZSh4ICogeCk7XG4gICAqICAgIH0sIE1hdGgucmFuZG9tKCkqMTAwMCArIDUwMCk7XG4gICAqICB9KSk7XG4gICAqIGNvbnNvbGUubG9nKGF3YWl0IGEudG9BcnJheSgpKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSB0cmFuc2Zvcm0gQSBmdW5jdGlvbiBtYXBwaW5nIGEgZGF0YXNldCBlbGVtZW50IHRvIGEgYFByb21pc2VgIGZvciBhXG4gICAqICAgdHJhbnNmb3JtZWQgZGF0YXNldCBlbGVtZW50LiAgVGhpcyB0cmFuc2Zvcm0gaXMgcmVzcG9uc2libGUgZm9yIGRpc3Bvc2luZ1xuICAgKiAgIGFueSBpbnRlcm1lZGlhdGUgYFRlbnNvcmBzLCBpLmUuIGJ5IHdyYXBwaW5nIGl0cyBjb21wdXRhdGlvbiBpblxuICAgKiAgIGB0Zi50aWR5KClgOyB0aGF0IGNhbm5vdCBiZSBhdXRvbWF0ZWQgaGVyZSAoYXMgaXQgaXMgaW4gdGhlIHN5bmNocm9ub3VzXG4gICAqICAgYG1hcCgpYCBjYXNlKS5cbiAgICpcbiAgICogQHJldHVybnMgQSBgRGF0YXNldGAgb2YgdHJhbnNmb3JtZWQgZWxlbWVudHMuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdEYXRhJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgbWFwQXN5bmM8TyBleHRlbmRzIHRmLlRlbnNvckNvbnRhaW5lcj4odHJhbnNmb3JtOiAodmFsdWU6IFQpID0+IFByb21pc2U8Tz4pOlxuICAgICAgRGF0YXNldDxPPiB7XG4gICAgY29uc3QgYmFzZSA9IHRoaXM7XG4gICAgcmV0dXJuIGRhdGFzZXRGcm9tSXRlcmF0b3JGbihhc3luYyAoKSA9PiB7XG4gICAgICByZXR1cm4gKGF3YWl0IGJhc2UuaXRlcmF0b3IoKSkubWFwQXN5bmModHJhbnNmb3JtKTtcbiAgICB9LCB0aGlzLnNpemUpO1xuICB9XG5cbiAgLyoqXG4gICAqICBDcmVhdGVzIGEgYERhdGFzZXRgIHRoYXQgcHJlZmV0Y2hlcyBlbGVtZW50cyBmcm9tIHRoaXMgZGF0YXNldC5cbiAgICpcbiAgICogQHBhcmFtIGJ1ZmZlclNpemU6IEFuIGludGVnZXIgc3BlY2lmeWluZyB0aGUgbnVtYmVyIG9mIGVsZW1lbnRzIHRvIGJlXG4gICAqICAgcHJlZmV0Y2hlZC5cbiAgICogQHJldHVybnMgQSBgRGF0YXNldGAuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdEYXRhJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgcHJlZmV0Y2goYnVmZmVyU2l6ZTogbnVtYmVyKTogRGF0YXNldDxUPiB7XG4gICAgaWYgKGJ1ZmZlclNpemUgPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IFJhbmdlRXJyb3IoXG4gICAgICAgICAgJ2BEYXRhc2V0LnByZWZldGNoKClgIHJlcXVpcmVzIGJ1ZmZlclNpemUgdG8gYmUgc3BlY2lmaWVkLicpO1xuICAgIH1cblxuICAgIGNvbnN0IGJhc2UgPSB0aGlzO1xuICAgIHJldHVybiBkYXRhc2V0RnJvbUl0ZXJhdG9yRm4oXG4gICAgICAgIGFzeW5jICgpID0+IChhd2FpdCBiYXNlLml0ZXJhdG9yKCkpLnByZWZldGNoKGJ1ZmZlclNpemUpLCB0aGlzLnNpemUpO1xuICB9XG5cbiAgLyoqXG4gICAqIFJlcGVhdHMgdGhpcyBkYXRhc2V0IGBjb3VudGAgdGltZXMuXG4gICAqXG4gICAqIE5PVEU6IElmIHRoaXMgZGF0YXNldCBpcyBhIGZ1bmN0aW9uIG9mIGdsb2JhbCBzdGF0ZSAoZS5nLiBhIHJhbmRvbSBudW1iZXJcbiAgICogZ2VuZXJhdG9yKSwgdGhlbiBkaWZmZXJlbnQgcmVwZXRpdGlvbnMgbWF5IHByb2R1Y2UgZGlmZmVyZW50IGVsZW1lbnRzLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBhID0gdGYuZGF0YS5hcnJheShbMSwgMiwgM10pLnJlcGVhdCgzKTtcbiAgICogYXdhaXQgYS5mb3JFYWNoQXN5bmMoZSA9PiBjb25zb2xlLmxvZyhlKSk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0gY291bnQ6IChPcHRpb25hbCkgQW4gaW50ZWdlciwgcmVwcmVzZW50aW5nIHRoZSBudW1iZXIgb2YgdGltZXNcbiAgICogICB0aGUgZGF0YXNldCBzaG91bGQgYmUgcmVwZWF0ZWQuIFRoZSBkZWZhdWx0IGJlaGF2aW9yIChpZiBgY291bnRgIGlzXG4gICAqICAgYHVuZGVmaW5lZGAgb3IgbmVnYXRpdmUpIGlzIGZvciB0aGUgZGF0YXNldCBiZSByZXBlYXRlZCBpbmRlZmluaXRlbHkuXG4gICAqIEByZXR1cm5zIEEgYERhdGFzZXRgLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIHJlcGVhdChjb3VudD86IG51bWJlcik6IERhdGFzZXQ8VD4ge1xuICAgIGNvbnN0IGJhc2UgPSB0aGlzO1xuICAgIGxldCBzaXplO1xuICAgIGlmICh0aGlzLnNpemUgIT0gbnVsbCAmJiBjb3VudCA+IDApIHtcbiAgICAgIC8vIElmIHRoaXMgZGF0YXNldCBoYXMgc2l6ZSBhbmQgY291bnQgaXMgcG9zaXRpdmUsIG5ldyBzaXplIGlzIGN1cnJlbnRcbiAgICAgIC8vIHNpemUgbXVsdGlwbHkgY291bnQuIFRoaXMgYWxzbyBjb3ZlcnMgdGhlIGNhc2UgdGhhdCBjdXJyZW50IHNpemUgaXNcbiAgICAgIC8vIGluZmluaXR5LlxuICAgICAgc2l6ZSA9IHRoaXMuc2l6ZSAqIGNvdW50O1xuICAgIH0gZWxzZSBpZiAoY291bnQgPT09IDApIHtcbiAgICAgIC8vIElmIGNvdW50IGlzIDAsIG5ldyBzaXplIGlzIDAuXG4gICAgICBzaXplID0gMDtcbiAgICB9IGVsc2UgaWYgKHRoaXMuc2l6ZSAhPSBudWxsICYmIChjb3VudCA9PT0gdW5kZWZpbmVkIHx8IGNvdW50IDwgMCkpIHtcbiAgICAgIC8vIElmIHRoaXMgZGF0YXNldCBoYXMgc2l6ZSBhbmQgY291bnQgaXMgdW5kZWZpbmVkIG9yIG5lZ2F0aXZlLCB0aGVcbiAgICAgIC8vIGRhdGFzZXQgd2lsbCBiZSByZXBlYXRlZCBpbmRlZmluaXRlbHkgYW5kIG5ldyBzaXplIGlzIGluZmluaXR5LlxuICAgICAgc2l6ZSA9IEluZmluaXR5O1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBJZiB0aGUgc2l6ZSBvZiB0aGlzIGRhdGFzZXQgaXMgbnVsbCwgdGhlIG5ldyBkYXRhc2V0J3Mgc2l6ZSBpcyBudWxsLlxuICAgICAgc2l6ZSA9IG51bGw7XG4gICAgfVxuICAgIHJldHVybiBkYXRhc2V0RnJvbUl0ZXJhdG9yRm4oYXN5bmMgKCkgPT4ge1xuICAgICAgY29uc3QgaXRlcmF0b3JJdGVyYXRvciA9IGl0ZXJhdG9yRnJvbUZ1bmN0aW9uKFxuICAgICAgICAgIGFzeW5jICgpID0+ICh7dmFsdWU6IGF3YWl0IGJhc2UuaXRlcmF0b3IoKSwgZG9uZTogZmFsc2V9KSk7XG4gICAgICByZXR1cm4gaXRlcmF0b3JGcm9tQ29uY2F0ZW5hdGVkKGl0ZXJhdG9ySXRlcmF0b3IudGFrZShjb3VudCkpO1xuICAgIH0sIHNpemUpO1xuICB9XG5cbiAgLyoqXG4gICAqIENyZWF0ZXMgYSBgRGF0YXNldGAgdGhhdCBza2lwcyBgY291bnRgIGluaXRpYWwgZWxlbWVudHMgZnJvbSB0aGlzIGRhdGFzZXQuXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IGEgPSB0Zi5kYXRhLmFycmF5KFsxLCAyLCAzLCA0LCA1LCA2XSkuc2tpcCgzKTtcbiAgICogYXdhaXQgYS5mb3JFYWNoQXN5bmMoZSA9PiBjb25zb2xlLmxvZyhlKSk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0gY291bnQ6IFRoZSBudW1iZXIgb2YgZWxlbWVudHMgb2YgdGhpcyBkYXRhc2V0IHRoYXQgc2hvdWxkIGJlIHNraXBwZWRcbiAgICogICB0byBmb3JtIHRoZSBuZXcgZGF0YXNldC4gIElmIGBjb3VudGAgaXMgZ3JlYXRlciB0aGFuIHRoZSBzaXplIG9mIHRoaXNcbiAgICogICBkYXRhc2V0LCB0aGUgbmV3IGRhdGFzZXQgd2lsbCBjb250YWluIG5vIGVsZW1lbnRzLiAgSWYgYGNvdW50YFxuICAgKiAgIGlzIGB1bmRlZmluZWRgIG9yIG5lZ2F0aXZlLCBza2lwcyB0aGUgZW50aXJlIGRhdGFzZXQuXG4gICAqXG4gICAqIEByZXR1cm5zIEEgYERhdGFzZXRgLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIHNraXAoY291bnQ6IG51bWJlcik6IERhdGFzZXQ8VD4ge1xuICAgIGNvbnN0IGJhc2UgPSB0aGlzO1xuICAgIGxldCBzaXplO1xuICAgIGlmICh0aGlzLnNpemUgIT0gbnVsbCAmJiBjb3VudCA+PSAwICYmIHRoaXMuc2l6ZSA+PSBjb3VudCkge1xuICAgICAgLy8gSWYgdGhlIHNpemUgb2YgdGhpcyBkYXRhc2V0IGlzIGdyZWF0ZXIgdGhhbiBjb3VudCwgdGhlIG5ldyBkYXRhc2V0J3NcbiAgICAgIC8vIHNpemUgaXMgY3VycmVudCBzaXplIG1pbnVzIHNraXBwZWQgc2l6ZS5UaGlzIGFsc28gY292ZXJzIHRoZSBjYXNlIHRoYXRcbiAgICAgIC8vIGN1cnJlbnQgc2l6ZSBpcyBpbmZpbml0eS5cbiAgICAgIHNpemUgPSB0aGlzLnNpemUgLSBjb3VudDtcbiAgICB9IGVsc2UgaWYgKFxuICAgICAgICB0aGlzLnNpemUgIT0gbnVsbCAmJlxuICAgICAgICAodGhpcy5zaXplIDwgY291bnQgfHwgY291bnQgPT09IHVuZGVmaW5lZCB8fCBjb3VudCA8IDApKSB7XG4gICAgICAvLyBJZiB0aGUgc2l6ZSBvZiB0aGlzIGRhdGFzZXQgaXMgc21hbGxlciB0aGFuIGNvdW50LCBvciBjb3VudCBpc1xuICAgICAgLy8gdW5kZWZpbmVkIG9yIG5lZ2F0aXZlLCBza2lwcyB0aGUgZW50aXJlIGRhdGFzZXQgYW5kIHRoZSBuZXcgc2l6ZSBpcyAwLlxuICAgICAgc2l6ZSA9IDA7XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIElmIHRoZSBzaXplIG9mIHRoaXMgZGF0YXNldCBpcyBudWxsLCB0aGUgbmV3IGRhdGFzZXQncyBzaXplIGlzIG51bGwuXG4gICAgICBzaXplID0gbnVsbDtcbiAgICB9XG4gICAgcmV0dXJuIGRhdGFzZXRGcm9tSXRlcmF0b3JGbihcbiAgICAgICAgYXN5bmMgKCkgPT4gKGF3YWl0IGJhc2UuaXRlcmF0b3IoKSkuc2tpcChjb3VudCksIHNpemUpO1xuICB9XG5cbiAgLy8gVE9ETyhzb2VyZ2VsKTogZGVlcCBzaGFyZGVkIHNodWZmbGUsIHdoZXJlIHN1cHBvcnRlZFxuXG4gIHN0YXRpYyByZWFkb25seSBNQVhfQlVGRkVSX1NJWkUgPSAxMDAwMDtcblxuICAvKipcbiAgICogUHNldWRvcmFuZG9tbHkgc2h1ZmZsZXMgdGhlIGVsZW1lbnRzIG9mIHRoaXMgZGF0YXNldC4gVGhpcyBpcyBkb25lIGluIGFcbiAgICogc3RyZWFtaW5nIG1hbm5lciwgYnkgc2FtcGxpbmcgZnJvbSBhIGdpdmVuIG51bWJlciBvZiBwcmVmZXRjaGVkIGVsZW1lbnRzLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBhID0gdGYuZGF0YS5hcnJheShbMSwgMiwgMywgNCwgNSwgNl0pLnNodWZmbGUoMyk7XG4gICAqIGF3YWl0IGEuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coZSkpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIGJ1ZmZlclNpemU6IEFuIGludGVnZXIgc3BlY2lmeWluZyB0aGUgbnVtYmVyIG9mIGVsZW1lbnRzIGZyb20gdGhpc1xuICAgKiAgIGRhdGFzZXQgZnJvbSB3aGljaCB0aGUgbmV3IGRhdGFzZXQgd2lsbCBzYW1wbGUuXG4gICAqIEBwYXJhbSBzZWVkOiAoT3B0aW9uYWwpIEFuIGludGVnZXIgc3BlY2lmeWluZyB0aGUgcmFuZG9tIHNlZWQgdGhhdCB3aWxsXG4gICAqICAgYmUgdXNlZCB0byBjcmVhdGUgdGhlIGRpc3RyaWJ1dGlvbi5cbiAgICogQHBhcmFtIHJlc2h1ZmZsZUVhY2hJdGVyYXRpb246IChPcHRpb25hbCkgQSBib29sZWFuLCB3aGljaCBpZiB0cnVlXG4gICAqICAgaW5kaWNhdGVzIHRoYXQgdGhlIGRhdGFzZXQgc2hvdWxkIGJlIHBzZXVkb3JhbmRvbWx5IHJlc2h1ZmZsZWQgZWFjaCB0aW1lXG4gICAqICAgaXQgaXMgaXRlcmF0ZWQgb3Zlci4gSWYgZmFsc2UsIGVsZW1lbnRzIHdpbGwgYmUgcmV0dXJuZWQgaW4gdGhlIHNhbWVcbiAgICogICBzaHVmZmxlZCBvcmRlciBvbiBlYWNoIGl0ZXJhdGlvbi4gKERlZmF1bHRzIHRvIGB0cnVlYC4pXG4gICAqIEByZXR1cm5zIEEgYERhdGFzZXRgLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIHNodWZmbGUoYnVmZmVyU2l6ZTogbnVtYmVyLCBzZWVkPzogc3RyaW5nLCByZXNodWZmbGVFYWNoSXRlcmF0aW9uID0gdHJ1ZSk6XG4gICAgICBEYXRhc2V0PFQ+IHtcbiAgICBpZiAoYnVmZmVyU2l6ZSA9PSBudWxsIHx8IGJ1ZmZlclNpemUgPCAwKSB7XG4gICAgICBpZiAodGhpcy5zaXplID09IG51bGwpIHtcbiAgICAgICAgdGhyb3cgbmV3IFJhbmdlRXJyb3IoXG4gICAgICAgICAgICAnYERhdGFzZXQuc2h1ZmZsZSgpYCByZXF1aXJlcyBidWZmZXJTaXplIHRvIGJlIHNwZWNpZmllZC4nKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBSYW5nZUVycm9yKFxuICAgICAgICAgICAgJ2BEYXRhc2V0LnNodWZmbGUoKWAgcmVxdWlyZXMgYnVmZmVyU2l6ZSB0byBiZSBzcGVjaWZpZWQuICAnICtcbiAgICAgICAgICAgICdJZiB5b3VyIGRhdGEgZml0cyBpbiBtYWluIG1lbW9yeSAoZm9yIHJlZ3VsYXIgSlMgb2JqZWN0cyksICcgK1xuICAgICAgICAgICAgJ2FuZC9vciBHUFUgbWVtb3J5IChmb3IgYHRmLlRlbnNvcmBzKSwgY29uc2lkZXIgc2V0dGluZyAnICtcbiAgICAgICAgICAgIGBidWZmZXJTaXplIHRvIHRoZSBkYXRhc2V0IHNpemUgKCR7dGhpcy5zaXplfSBlbGVtZW50cylgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgY29uc3QgYmFzZSA9IHRoaXM7XG4gICAgY29uc3QgcmFuZG9tID0gc2VlZHJhbmRvbS5hbGVhKHNlZWQgfHwgdGYudXRpbC5ub3coKS50b1N0cmluZygpKTtcbiAgICByZXR1cm4gZGF0YXNldEZyb21JdGVyYXRvckZuKGFzeW5jICgpID0+IHtcbiAgICAgIGxldCBzZWVkMiA9IHJhbmRvbS5pbnQzMigpO1xuICAgICAgaWYgKHJlc2h1ZmZsZUVhY2hJdGVyYXRpb24pIHtcbiAgICAgICAgc2VlZDIgKz0gcmFuZG9tLmludDMyKCk7XG4gICAgICB9XG4gICAgICByZXR1cm4gKGF3YWl0IGJhc2UuaXRlcmF0b3IoKSkuc2h1ZmZsZShidWZmZXJTaXplLCBzZWVkMi50b1N0cmluZygpKTtcbiAgICB9LCB0aGlzLnNpemUpO1xuICB9XG5cbiAgLyoqXG4gICAqIENyZWF0ZXMgYSBgRGF0YXNldGAgd2l0aCBhdCBtb3N0IGBjb3VudGAgaW5pdGlhbCBlbGVtZW50cyBmcm9tIHRoaXNcbiAgICogZGF0YXNldC5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgYSA9IHRmLmRhdGEuYXJyYXkoWzEsIDIsIDMsIDQsIDUsIDZdKS50YWtlKDMpO1xuICAgKiBhd2FpdCBhLmZvckVhY2hBc3luYyhlID0+IGNvbnNvbGUubG9nKGUpKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSBjb3VudDogVGhlIG51bWJlciBvZiBlbGVtZW50cyBvZiB0aGlzIGRhdGFzZXQgdGhhdCBzaG91bGQgYmUgdGFrZW5cbiAgICogICB0byBmb3JtIHRoZSBuZXcgZGF0YXNldC4gIElmIGBjb3VudGAgaXMgYHVuZGVmaW5lZGAgb3IgbmVnYXRpdmUsIG9yIGlmXG4gICAqICAgYGNvdW50YCBpcyBncmVhdGVyIHRoYW4gdGhlIHNpemUgb2YgdGhpcyBkYXRhc2V0LCB0aGUgbmV3IGRhdGFzZXQgd2lsbFxuICAgKiAgIGNvbnRhaW4gYWxsIGVsZW1lbnRzIG9mIHRoaXMgZGF0YXNldC5cbiAgICogQHJldHVybnMgQSBgRGF0YXNldGAuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdEYXRhJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgdGFrZShjb3VudDogbnVtYmVyKTogRGF0YXNldDxUPiB7XG4gICAgY29uc3QgYmFzZSA9IHRoaXM7XG4gICAgbGV0IHNpemU7XG4gICAgaWYgKHRoaXMuc2l6ZSAhPSBudWxsICYmIHRoaXMuc2l6ZSA+IGNvdW50KSB7XG4gICAgICAvLyBJZiB0aGUgc2l6ZSBvZiB0aGlzIGRhdGFzZXQgaXMgZ3JlYXRlciB0aGFuIGNvdW50LCB0aGUgbmV3IGRhdGFzZXQnc1xuICAgICAgLy8gc2l6ZSBpcyBjb3VudC5cbiAgICAgIHNpemUgPSBjb3VudDtcbiAgICB9IGVsc2UgaWYgKHRoaXMuc2l6ZSAhPSBudWxsICYmIHRoaXMuc2l6ZSA8PSBjb3VudCkge1xuICAgICAgLy8gSWYgdGhlIHNpemUgb2YgdGhpcyBkYXRhc2V0IGlzIGVxdWFsIG9yIHNtYWxsZXIgdGhhbiBjb3VudCwgdGhlIG5ld1xuICAgICAgLy8gZGF0YXNldCdzIHNpemUgaXMgdGhlIHNpemUgb2YgdGhpcyBkYXRhc2V0LlxuICAgICAgc2l6ZSA9IHRoaXMuc2l6ZTtcbiAgICB9IGVsc2Uge1xuICAgICAgLy8gSWYgdGhlIHNpemUgb2YgdGhpcyBkYXRhc2V0IGlzIG51bGwsIHRoZSBuZXcgZGF0YXNldCdzIHNpemUgaXMgbnVsbC5cbiAgICAgIHNpemUgPSBudWxsO1xuICAgIH1cbiAgICByZXR1cm4gZGF0YXNldEZyb21JdGVyYXRvckZuKFxuICAgICAgICBhc3luYyAoKSA9PiAoYXdhaXQgYmFzZS5pdGVyYXRvcigpKS50YWtlKGNvdW50KSwgc2l6ZSk7XG4gIH1cblxuICAvKipcbiAgICogQ29sbGVjdCBhbGwgZWxlbWVudHMgb2YgdGhpcyBkYXRhc2V0IGludG8gYW4gYXJyYXkuXG4gICAqXG4gICAqIE9idmlvdXNseSB0aGlzIHdpbGwgc3VjY2VlZCBvbmx5IGZvciBzbWFsbCBkYXRhc2V0cyB0aGF0IGZpdCBpbiBtZW1vcnkuXG4gICAqIFVzZWZ1bCBmb3IgdGVzdGluZyBhbmQgZ2VuZXJhbGx5IHNob3VsZCBiZSBhdm9pZGVkIGlmIHBvc3NpYmxlLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBhID0gdGYuZGF0YS5hcnJheShbMSwgMiwgMywgNCwgNSwgNl0pO1xuICAgKiBjb25zb2xlLmxvZyhhd2FpdCBhLnRvQXJyYXkoKSk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcmV0dXJucyBBIFByb21pc2UgZm9yIGFuIGFycmF5IG9mIGVsZW1lbnRzLCB3aGljaCB3aWxsIHJlc29sdmVcbiAgICogICB3aGVuIGEgbmV3IHN0cmVhbSBoYXMgYmVlbiBvYnRhaW5lZCBhbmQgZnVsbHkgY29uc3VtZWQuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdEYXRhJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgYXN5bmMgdG9BcnJheSgpIHtcbiAgICBpZiAodGhpcy5zaXplID09PSBJbmZpbml0eSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdDYW4gbm90IGNvbnZlcnQgaW5maW5pdGUgZGF0YSBzdHJlYW0gdG8gYXJyYXkuJyk7XG4gICAgfVxuICAgIHJldHVybiAoYXdhaXQgdGhpcy5pdGVyYXRvcigpKS50b0FycmF5KCk7XG4gIH1cblxuICAvKipcbiAgICogQ29sbGVjdCBhbGwgZWxlbWVudHMgb2YgdGhpcyBkYXRhc2V0IGludG8gYW4gYXJyYXkgd2l0aCBwcmVmZXRjaGluZyAxMDBcbiAgICogZWxlbWVudHMuIFRoaXMgaXMgdXNlZnVsIGZvciB0ZXN0aW5nLCBiZWNhdXNlIHRoZSBwcmVmZXRjaCBjaGFuZ2VzIHRoZVxuICAgKiBvcmRlciBpbiB3aGljaCB0aGUgUHJvbWlzZXMgYXJlIHJlc29sdmVkIGFsb25nIHRoZSBwcm9jZXNzaW5nIHBpcGVsaW5lLlxuICAgKiBUaGlzIG1heSBoZWxwIGV4cG9zZSBidWdzIHdoZXJlIHJlc3VsdHMgYXJlIGRlcGVuZGVudCBvbiB0aGUgb3JkZXIgb2ZcbiAgICogUHJvbWlzZSByZXNvbHV0aW9uIHJhdGhlciB0aGFuIG9uIHRoZSBsb2dpY2FsIG9yZGVyIG9mIHRoZSBzdHJlYW0gKGkuZS4sXG4gICAqIGR1ZSB0byBoaWRkZW4gbXV0YWJsZSBzdGF0ZSkuXG4gICAqXG4gICAqIEByZXR1cm5zIEEgUHJvbWlzZSBmb3IgYW4gYXJyYXkgb2YgZWxlbWVudHMsIHdoaWNoIHdpbGwgcmVzb2x2ZVxuICAgKiAgIHdoZW4gYSBuZXcgc3RyZWFtIGhhcyBiZWVuIG9idGFpbmVkIGFuZCBmdWxseSBjb25zdW1lZC5cbiAgICovXG4gIGFzeW5jIHRvQXJyYXlGb3JUZXN0KCkge1xuICAgIGlmICh0aGlzLnNpemUgPT09IEluZmluaXR5KSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0NhbiBub3QgY29udmVydCBpbmZpbml0ZSBkYXRhIHN0cmVhbSB0byBhcnJheS4nKTtcbiAgICB9XG4gICAgcmV0dXJuIChhd2FpdCB0aGlzLml0ZXJhdG9yKCkpLnRvQXJyYXlGb3JUZXN0KCk7XG4gIH1cbn1cblxuLyoqXG4gKiBDcmVhdGUgYSBgRGF0YXNldGAgZGVmaW5lZCBieSBhIHByb3ZpZGVkIGl0ZXJhdG9yKCkgZnVuY3Rpb24uXG4gKlxuICogYGBganNcbiAqIGxldCBpID0gLTE7XG4gKiBjb25zdCBmdW5jID0gKCkgPT5cbiAqICAgICsraSA8IDUgPyB7dmFsdWU6IGksIGRvbmU6IGZhbHNlfSA6IHt2YWx1ZTogbnVsbCwgZG9uZTogdHJ1ZX07XG4gKiBjb25zdCBpdGVyID0gdGYuZGF0YS5pdGVyYXRvckZyb21GdW5jdGlvbihmdW5jKTtcbiAqIGNvbnN0IGRzID0gdGYuZGF0YS5kYXRhc2V0RnJvbUl0ZXJhdG9yRm4oaXRlcik7XG4gKiBhd2FpdCBkcy5mb3JFYWNoQXN5bmMoZSA9PiBjb25zb2xlLmxvZyhlKSk7XG4gKiBgYGBcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGRhdGFzZXRGcm9tSXRlcmF0b3JGbjxUIGV4dGVuZHMgdGYuVGVuc29yQ29udGFpbmVyPihcbiAgICBpdGVyYXRvckZuOiAoKSA9PiBQcm9taXNlPExhenlJdGVyYXRvcjxUPj4sXG4gICAgc2l6ZTogbnVtYmVyID0gbnVsbCk6IERhdGFzZXQ8VD4ge1xuICByZXR1cm4gbmV3IGNsYXNzIGV4dGVuZHMgRGF0YXNldDxUPiB7XG4gICAgb3ZlcnJpZGUgc2l6ZSA9IHNpemU7XG5cbiAgICAvKlxuICAgICAqIFByb3ZpZGUgYSBuZXcgc3RyZWFtIG9mIGVsZW1lbnRzLiAgTm90ZSB0aGlzIHdpbGwgYWxzbyBzdGFydCBuZXcgc3RyZWFtc1xuICAgICAqIGZyb20gYW55IHVuZGVybHlpbmcgYERhdGFzZXRgcy5cbiAgICAgKi9cbiAgICBhc3luYyBpdGVyYXRvcigpOiBQcm9taXNlPExhenlJdGVyYXRvcjxUPj4ge1xuICAgICAgcmV0dXJuIGl0ZXJhdG9yRm4oKTtcbiAgICB9XG4gIH1cbiAgKCk7XG59XG5cbi8qKlxuICogQ3JlYXRlIGEgYERhdGFzZXRgIGZyb20gYW4gYXJyYXkgb2YgZWxlbWVudHMuXG4gKlxuICogQ3JlYXRlIGEgRGF0YXNldCBmcm9tIGFuIGFycmF5IG9mIG9iamVjdHM6XG4gKiBgYGBqc1xuICogY29uc3QgYSA9IHRmLmRhdGEuYXJyYXkoW3snaXRlbSc6IDF9LCB7J2l0ZW0nOiAyfSwgeydpdGVtJzogM31dKTtcbiAqIGF3YWl0IGEuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coZSkpO1xuICogYGBgXG4gKlxuICogQ3JlYXRlIGEgRGF0YXNldCBmcm9tIGFuIGFycmF5IG9mIG51bWJlcnM6XG4gKiBgYGBqc1xuICogY29uc3QgYSA9IHRmLmRhdGEuYXJyYXkoWzQsIDUsIDZdKTtcbiAqIGF3YWl0IGEuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coZSkpO1xuICogYGBgXG4gKiBAcGFyYW0gaXRlbXMgQW4gYXJyYXkgb2YgZWxlbWVudHMgdGhhdCB3aWxsIGJlIHBhcnNlZCBhcyBpdGVtcyBpbiBhIGRhdGFzZXQuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0RhdGEnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nLCBuYW1lc3BhY2U6ICdkYXRhJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGFycmF5PFQgZXh0ZW5kcyB0Zi5UZW5zb3JDb250YWluZXI+KGl0ZW1zOiBUW10pOiBEYXRhc2V0PFQ+IHtcbiAgcmV0dXJuIGRhdGFzZXRGcm9tSXRlcmF0b3JGbihcbiAgICAgIGFzeW5jICgpID0+IGl0ZXJhdG9yRnJvbUl0ZW1zKGl0ZW1zKSwgaXRlbXMubGVuZ3RoKTtcbn1cblxuLyoqXG4gKiBDcmVhdGUgYSBgRGF0YXNldGAgYnkgemlwcGluZyB0b2dldGhlciBhbiBhcnJheSwgZGljdCwgb3IgbmVzdGVkXG4gKiBzdHJ1Y3R1cmUgb2YgYERhdGFzZXRgcyAoYW5kIHBlcmhhcHMgYWRkaXRpb25hbCBjb25zdGFudHMpLlxuICogVGhlIHVuZGVybHlpbmcgZGF0YXNldHMgbXVzdCBwcm92aWRlIGVsZW1lbnRzIGluIGEgY29uc2lzdGVudCBvcmRlciBzdWNoIHRoYXRcbiAqIHRoZXkgY29ycmVzcG9uZC5cbiAqXG4gKiBUaGUgbnVtYmVyIG9mIGVsZW1lbnRzIGluIHRoZSByZXN1bHRpbmcgZGF0YXNldCBpcyB0aGUgc2FtZSBhcyB0aGUgc2l6ZSBvZlxuICogdGhlIHNtYWxsZXN0IGRhdGFzZXQgaW4gZGF0YXNldHMuXG4gKlxuICogVGhlIG5lc3RlZCBzdHJ1Y3R1cmUgb2YgdGhlIGBkYXRhc2V0c2AgYXJndW1lbnQgZGV0ZXJtaW5lcyB0aGVcbiAqIHN0cnVjdHVyZSBvZiBlbGVtZW50cyBpbiB0aGUgcmVzdWx0aW5nIGl0ZXJhdG9yLlxuICpcbiAqIE5vdGUgdGhpcyBtZWFucyB0aGF0LCBnaXZlbiBhbiBhcnJheSBvZiB0d28gZGF0YXNldHMgdGhhdCBwcm9kdWNlIGRpY3RcbiAqIGVsZW1lbnRzLCB0aGUgcmVzdWx0IGlzIGEgZGF0YXNldCB0aGF0IHByb2R1Y2VzIGVsZW1lbnRzIHRoYXQgYXJlIGFycmF5c1xuICogb2YgdHdvIGRpY3RzOlxuICpcbiAqIFppcCBhbiBhcnJheSBvZiBkYXRhc2V0czpcbiAqIGBgYGpzXG4gKiBjb25zb2xlLmxvZygnWmlwIHR3byBkYXRhc2V0cyBvZiBvYmplY3RzOicpO1xuICogY29uc3QgZHMxID0gdGYuZGF0YS5hcnJheShbe2E6IDF9LCB7YTogMn0sIHthOiAzfV0pO1xuICogY29uc3QgZHMyID0gdGYuZGF0YS5hcnJheShbe2I6IDR9LCB7YjogNX0sIHtiOiA2fV0pO1xuICogY29uc3QgZHMzID0gdGYuZGF0YS56aXAoW2RzMSwgZHMyXSk7XG4gKiBhd2FpdCBkczMuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coSlNPTi5zdHJpbmdpZnkoZSkpKTtcbiAqXG4gKiAvLyBJZiB0aGUgZ29hbCBpcyB0byBtZXJnZSB0aGUgZGljdHMgaW4gb3JkZXIgdG8gcHJvZHVjZSBlbGVtZW50cyBsaWtlXG4gKiAvLyB7YTogLi4uLCBiOiAuLi59LCB0aGlzIHJlcXVpcmVzIGEgc2Vjb25kIHN0ZXAgc3VjaCBhczpcbiAqIGNvbnNvbGUubG9nKCdNZXJnZSB0aGUgb2JqZWN0czonKTtcbiAqIGNvbnN0IGRzNCA9IGRzMy5tYXAoeCA9PiB7cmV0dXJuIHthOiB4WzBdLmEsIGI6IHhbMV0uYn19KTtcbiAqIGF3YWl0IGRzNC5mb3JFYWNoQXN5bmMoZSA9PiBjb25zb2xlLmxvZyhlKSk7XG4gKiBgYGBcbiAqXG4gKiBaaXAgYSBkaWN0IG9mIGRhdGFzZXRzOlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi5kYXRhLmFycmF5KFt7YTogMX0sIHthOiAyfSwge2E6IDN9XSk7XG4gKiBjb25zdCBiID0gdGYuZGF0YS5hcnJheShbe2I6IDR9LCB7YjogNX0sIHtiOiA2fV0pO1xuICogY29uc3QgYyA9IHRmLmRhdGEuemlwKHtjOiBhLCBkOiBifSk7XG4gKiBhd2FpdCBjLmZvckVhY2hBc3luYyhlID0+IGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KGUpKSk7XG4gKiBgYGBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdPcGVyYXRpb25zJywgbmFtZXNwYWNlOiAnZGF0YSd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiB6aXA8TyBleHRlbmRzIHRmLlRlbnNvckNvbnRhaW5lcj4oZGF0YXNldHM6IERhdGFzZXRDb250YWluZXIpOlxuICAgIERhdGFzZXQ8Tz4ge1xuICAvLyBtYW51YWxseSB0eXBlLWNoZWNrIHRoZSBhcmd1bWVudCBmb3IgSlMgdXNlcnNcbiAgaWYgKCFpc0l0ZXJhYmxlKGRhdGFzZXRzKSkge1xuICAgIHRocm93IG5ldyBFcnJvcignVGhlIGFyZ3VtZW50IHRvIHppcCgpIG11c3QgYmUgYW4gb2JqZWN0IG9yIGFycmF5LicpO1xuICB9XG4gIGxldCBzaXplO1xuICBpZiAoQXJyYXkuaXNBcnJheShkYXRhc2V0cykpIHtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGRhdGFzZXRzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBzaXplID0gc2l6ZSA9PSBudWxsID8gKGRhdGFzZXRzW2ldIGFzIERhdGFzZXQ8Tz4pLnNpemUgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIE1hdGgubWluKHNpemUsIChkYXRhc2V0c1tpXSBhcyBEYXRhc2V0PE8+KS5zaXplKTtcbiAgICB9XG4gIH0gZWxzZSBpZiAoZGF0YXNldHMgaW5zdGFuY2VvZiBPYmplY3QpIHtcbiAgICBmb3IgKGNvbnN0IGRzIGluIGRhdGFzZXRzKSB7XG4gICAgICBzaXplID0gc2l6ZSA9PSBudWxsID8gKGRhdGFzZXRzW2RzXSBhcyBEYXRhc2V0PE8+KS5zaXplIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBNYXRoLm1pbihzaXplLCAoZGF0YXNldHNbZHNdIGFzIERhdGFzZXQ8Tz4pLnNpemUpO1xuICAgIH1cbiAgfVxuICByZXR1cm4gZGF0YXNldEZyb21JdGVyYXRvckZuPE8+KGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBzdHJlYW1zID0gYXdhaXQgZGVlcE1hcEFuZEF3YWl0QWxsKGRhdGFzZXRzLCBkID0+IHtcbiAgICAgIGlmIChkIGluc3RhbmNlb2YgRGF0YXNldCkge1xuICAgICAgICByZXR1cm4ge3ZhbHVlOiBkLml0ZXJhdG9yKCksIHJlY3Vyc2U6IGZhbHNlfTtcbiAgICAgIH0gZWxzZSBpZiAoaXNJdGVyYWJsZShkKSkge1xuICAgICAgICByZXR1cm4ge3ZhbHVlOiBudWxsLCByZWN1cnNlOiB0cnVlfTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICdMZWF2ZXMgb2YgdGhlIHN0cnVjdHVyZSBwYXNzZWQgdG8gemlwKCkgbXVzdCBiZSBEYXRhc2V0cywgJyArXG4gICAgICAgICAgICAnbm90IHByaW1pdGl2ZXMuJyk7XG4gICAgICB9XG4gICAgfSk7XG4gICAgcmV0dXJuIGl0ZXJhdG9yRnJvbVppcHBlZDxPPihzdHJlYW1zLCBaaXBNaXNtYXRjaE1vZGUuU0hPUlRFU1QpO1xuICB9LCBzaXplKTtcbn1cblxuLyoqXG4gKiBBIHppcCBmdW5jdGlvbiBmb3IgdXNlIHdpdGggZGVlcFppcCwgcGFzc2VkIHZpYSB0aGUgY29sdW1uTWFqb3JCYXRjaCBjYWxsLlxuICpcbiAqIEFjY2VwdHMgYW4gYXJyYXkgb2YgaWRlbnRpY2FsbHktc3RydWN0dXJlZCBuZXN0ZWQgZWxlbWVudHMgYW5kIGVpdGhlciBiYXRjaGVzXG4gKiB0aGVtIChpZiB0aGV5IGFyZSBwcmltaXRpdmVzLCBudW1lcmljIGFycmF5cywgb3IgVGVuc29ycykgb3IgcmVxdWVzdHNcbiAqIHJlY3Vyc2lvbiAoaWYgbm90KS5cbiAqL1xuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuZnVuY3Rpb24gZGVlcEJhdGNoQ29uY2F0KHJvd3M6IGFueVtdKTogRGVlcE1hcFJlc3VsdCB7XG4gIGlmIChyb3dzID09PSBudWxsKSB7XG4gICAgcmV0dXJuIG51bGw7XG4gIH1cblxuICAvLyB1c2UgdGhlIGZpcnN0IGl0ZW0gdG8gZGVjaWRlIHdoZXRoZXIgdG8gcmVjdXJzZSBvciBiYXRjaCBoZXJlLlxuICBjb25zdCBleGFtcGxlUm93ID0gcm93c1swXTtcblxuICBpZiAoY2FuVGVuc29yaWZ5KGV4YW1wbGVSb3cpKSB7XG4gICAgLy8gcm93cyBpcyBhbiBhcnJheSBvZiBwcmltaXRpdmVzLCBUZW5zb3JzLCBvciBhcnJheXMuICBCYXRjaCB0aGVtLlxuICAgIGNvbnN0IHZhbHVlID0gYmF0Y2hDb25jYXQocm93cyk7XG4gICAgcmV0dXJuIHt2YWx1ZSwgcmVjdXJzZTogZmFsc2V9O1xuICB9XG5cbiAgLy8gdGhlIGV4YW1wbGUgcm93IGlzIGFuIG9iamVjdCwgc28gcmVjdXJzZSBpbnRvIGl0LlxuICByZXR1cm4ge3ZhbHVlOiBudWxsLCByZWN1cnNlOiB0cnVlfTtcbn1cblxuLyoqXG4gKiBBc3NlbWJsZXMgYSBsaXN0IG9mIHNhbWUtc2hhcGVkIG51bWJlcnMsIG51bWJlciBhcnJheXMsIG9yIFRlbnNvcnNcbiAqIGludG8gYSBzaW5nbGUgbmV3IFRlbnNvciB3aGVyZSBheGlzIDAgaXMgdGhlIGJhdGNoIGRpbWVuc2lvbi5cbiAqL1xuZnVuY3Rpb24gYmF0Y2hDb25jYXQ8VCBleHRlbmRzKFRlbnNvckxpa2UgfCB0Zi5UZW5zb3IpPihhcnJheXM6IFRbXSk6XG4gICAgdGYuVGVuc29yIHtcbiAgaWYgKGFycmF5cy5sZW5ndGggPT09IDApIHtcbiAgICAvLyBXZSBjYW4ndCByZXR1cm4gYW4gZW1wdHkgVGVuc29yIGJlY2F1c2Ugd2UgZG9uJ3Qga25vdyB0aGUgZWxlbWVudCBzaGFwZS5cbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0NhblxcJ3QgbWFrZSBhIGJhdGNoIG9mIHplcm8gZWxlbWVudHMuJyk7XG4gIH1cblxuICBpZiAoYXJyYXlzWzBdIGluc3RhbmNlb2YgdGYuVGVuc29yKSB7XG4gICAgLy8gSW5wdXQgaXMgYW4gYXJyYXkgb2YgVGVuc29yc1xuICAgIHJldHVybiB0Zi5zdGFjayhhcnJheXMgYXMgdGYuVGVuc29yW10pO1xuICB9IGVsc2Uge1xuICAgIC8vIElucHV0IGlzIGEgcG9zc2libHktbmVzdGVkIGFycmF5IG9mIG51bWJlcnMuXG4gICAgcmV0dXJuIHRmLnRlbnNvcihhcnJheXMgYXMgVGVuc29yTGlrZSk7XG4gIH1cbn1cbiJdfQ==