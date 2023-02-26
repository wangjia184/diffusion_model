/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
 * =============================================================================
 */
import * as util from '../util';
const NEW_AXIS = -2;
const SHRINK_AXIS = -1;
export function assertParamsValid(input, begin, size) {
    const inputRank = input.shape.length;
    util.assert(inputRank === begin.length, () => `Error in slice${inputRank}D: Length of begin ${begin} must ` +
        `match the rank of the array (${inputRank}).`);
    util.assert(inputRank === size.length, () => `Error in slice${inputRank}D: Length of size ${size} must ` +
        `match the rank of the array (${inputRank}).`);
    for (let i = 0; i < inputRank; ++i) {
        util.assert(begin[i] + size[i] <= input.shape[i], () => `Error in slice${inputRank}D: begin[${i}] + size[${i}] ` +
            `(${begin[i] + size[i]}) would overflow input.shape[${i}] (${input.shape[i]})`);
    }
}
/** Converts a binary mask to an array of axes. Used in stridedSlice(). */
export function maskToAxes(mask) {
    const axes = [];
    let axis = 0;
    while (mask > 0) {
        if (mask & 1) {
            axes.push(axis);
        }
        mask /= 2;
        axis++;
    }
    return axes;
}
/** Computes the output shape given the strided slice params. */
export function computeOutShape(begin, end, strides) {
    const size = [];
    for (let axis = 0; axis < begin.length; axis++) {
        size[axis] = Math.ceil((end[axis] - begin[axis]) / strides[axis]);
    }
    return size;
}
// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current stride value. Otherwise, insert.
export function stridesWithElidedDims(strides, ellipsisInsertionIndex, numElidedAxes, inputShape) {
    const newStrides = [...strides];
    for (let i = newStrides.length; i < inputShape.length; i++) {
        newStrides.push(1);
    }
    for (let i = 0; i < numElidedAxes; i++) {
        if (i === 0) {
            newStrides[ellipsisInsertionIndex] = 1;
        }
        else {
            newStrides.splice(ellipsisInsertionIndex, 0 /* num elements to delete */, 1 /* element to add */);
            newStrides.pop();
        }
    }
    return newStrides;
}
function unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, normalizedAxis) {
    if (normalizedAxis <= ellipsisInsertionIndex) {
        return normalizedAxis;
    }
    return normalizedAxis - (numElidedAxes - 1);
}
function getElidedAxes(numElidedAxes, ellipsisInsertionIndex) {
    const elidedAxes = [];
    for (let i = 0; i < numElidedAxes; i++) {
        elidedAxes.push(ellipsisInsertionIndex + i);
    }
    return elidedAxes;
}
// Normalize the start, end and strides.
export function getNormalizedAxes(inputShape, ellipsisAxes, numInterpolatedAxes, begin, end, strides, beginMask, endMask, ellipsisMask) {
    const inputRank = inputShape.length;
    let normalizedBegin = new Array(inputRank), normalizedEnd = new Array(inputRank), normalizedStrides = new Array(inputRank);
    if (ellipsisAxes.length && numInterpolatedAxes > 0) {
        const fullIndex = ellipsisAxes[0];
        // The ellipsis applies to the masked index as well as any dimensions
        // that are interpolated.
        const numElidedAxes = numInterpolatedAxes + 1;
        normalizedBegin = startIndicesWithElidedDims(beginMask, fullIndex, numElidedAxes, begin, inputShape);
        normalizedEnd = stopIndicesWithElidedDims(endMask, fullIndex, numElidedAxes, end, inputShape);
        normalizedStrides =
            stridesWithElidedDims(strides, fullIndex, numElidedAxes, inputShape);
    }
    else {
        for (let axis = 0; axis < inputRank; axis++) {
            normalizedBegin[axis] = startForAxis(beginMask, begin, strides, inputShape, axis, ellipsisMask);
            normalizedEnd[axis] =
                stopForAxis(endMask, end, strides, inputShape, axis, ellipsisMask);
            normalizedStrides[axis] = stridesForAxis(strides, axis, ellipsisMask);
        }
    }
    return {
        begin: normalizedBegin,
        end: normalizedEnd,
        strides: normalizedStrides
    };
}
// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current start value. Otherwise, insert.
export function startIndicesWithElidedDims(beginMask, ellipsisInsertionIndex, numElidedAxes, originalBegin, inputShape) {
    const newIndices = [...inputShape];
    const elidedAxes = getElidedAxes(numElidedAxes, ellipsisInsertionIndex);
    for (let axis = 0; axis < newIndices.length; axis++) {
        if (elidedAxes.indexOf(axis) > -1) {
            newIndices[axis] = 0;
        }
        else {
            const originalAxis = unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, axis);
            let originalValue = originalBegin[originalAxis];
            if (beginMask & 1 << originalAxis) {
                originalValue = 0;
            }
            newIndices[axis] = originalValue;
        }
    }
    return newIndices;
}
// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current stop value. Otherwise, insert.
export function stopIndicesWithElidedDims(endMask, ellipsisInsertionIndex, numElidedAxes, originalEnd, inputShape) {
    const newIndices = [...inputShape];
    const elidedAxes = getElidedAxes(numElidedAxes, ellipsisInsertionIndex);
    for (let axis = 0; axis < newIndices.length; axis++) {
        if (elidedAxes.indexOf(axis) > -1) {
            newIndices[axis] = Number.MAX_SAFE_INTEGER;
        }
        else {
            const originalAxis = unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, axis);
            let originalValue = originalEnd[originalAxis];
            if (endMask & 1 << originalAxis) {
                originalValue = Number.MAX_SAFE_INTEGER;
            }
            newIndices[axis] = originalValue;
        }
    }
    for (let i = 0; i < newIndices.length; i++) {
        // Handle negative indices
        const axisSize = inputShape[i];
        if (newIndices[i] < 0) {
            newIndices[i] += axisSize;
        }
        newIndices[i] = util.clamp(0, newIndices[i], inputShape[i]);
    }
    return newIndices;
}
export function stridesForAxis(strides, axis, ellipsisMask) {
    let stride = strides[axis];
    if (ellipsisMask & (1 << axis) || stride == null) {
        stride = 1;
    }
    return stride;
}
export function startForAxis(beginMask, startIndices, strides, inputShape, axis, ellipsisMask) {
    // Begin with the specified index
    let start = startIndices[axis];
    const stride = strides[axis] || 1;
    // Check the axis bit from right of masked axes, or the begin index is not set
    // for the axis.
    if (beginMask & 1 << axis || ellipsisMask & 1 << axis || start == null) {
        if (stride > 0) {
            // Forward iteration - use the first element. These values will get
            // clamped below (Note: We could have set them to 0 and axis_size-1, but
            // use lowest() and max() to maintain symmetry with StopForAxis())
            start = Number.MIN_SAFE_INTEGER;
        }
        else {
            // Backward iteration - use the last element.
            start = Number.MAX_SAFE_INTEGER;
        }
    }
    // Handle negative indices
    const axisSize = inputShape[axis];
    if (start < 0) {
        start += axisSize;
    }
    // Clamping
    start = util.clamp(0, start, axisSize - 1);
    return start;
}
export function stopForAxis(endMask, stopIndices, strides, inputShape, axis, ellipsisMask) {
    // Begin with the specified index
    let stop = stopIndices[axis];
    const stride = strides[axis] || 1;
    // Check the axis bit from right of masked axes, or if the stop index is not
    // set for this axis.
    if (endMask & (1 << axis) || ellipsisMask & (1 << axis) || stop == null) {
        if (stride > 0) {
            // Forward iteration - use the last element. These values will get
            // clamped below
            stop = Number.MAX_SAFE_INTEGER;
        }
        else {
            // Backward iteration - use the first element.
            stop = Number.MIN_SAFE_INTEGER;
        }
    }
    // Handle negative indices
    const axisSize = inputShape[axis];
    if (stop < 0) {
        stop += axisSize;
    }
    // Clamping
    // Because the end index points one past the last element, we need slightly
    // different clamping ranges depending on the direction.
    if (stride > 0) {
        // Forward iteration
        stop = util.clamp(0, stop, axisSize);
    }
    else {
        // Backward iteration
        stop = util.clamp(-1, stop, axisSize - 1);
    }
    return stop;
}
/**
 * Returns true if the slice occupies a continous set of elements in the
 * 'flat' space.
 */
export function isSliceContinous(shape, begin, size) {
    // Index of the first axis that has size > 1.
    let firstNonOneAxis = size.length;
    for (let i = 0; i < size.length; i++) {
        if (size[i] > 1) {
            firstNonOneAxis = i;
            break;
        }
    }
    for (let i = firstNonOneAxis + 1; i < size.length; i++) {
        if (begin[i] > 0 || size[i] !== shape[i]) {
            return false;
        }
    }
    return true;
}
export function computeFlatOffset(begin, strides) {
    let flatOffset = begin.length > 0 ? begin[begin.length - 1] : 1;
    for (let i = 0; i < begin.length - 1; i++) {
        flatOffset += begin[i] * strides[i];
    }
    return flatOffset;
}
export function parseSliceParams(x, begin, size) {
    // The following logic allows for more ergonomic calls.
    let begin_;
    const xRank = x.shape.length;
    if (typeof begin === 'number') {
        begin_ = [begin, ...new Array(xRank - 1).fill(0)];
    }
    else if (begin.length < xRank) {
        begin_ = begin.concat(new Array(xRank - begin.length).fill(0));
    }
    else {
        begin_ = begin.slice();
    }
    begin_.forEach(d => {
        util.assert(d !== -1, () => 'slice() does not support negative begin indexing.');
    });
    let size_;
    if (size == null) {
        size_ = new Array(xRank).fill(-1);
    }
    else if (typeof size === 'number') {
        size_ = [size, ...new Array(xRank - 1).fill(-1)];
    }
    else if (size.length < xRank) {
        size_ = size.concat(new Array(xRank - size.length).fill(-1));
    }
    else {
        size_ = size;
    }
    size_ = size_.map((d, i) => {
        if (d >= 0) {
            return d;
        }
        else {
            util.assert(d === -1, () => `Negative size values should be exactly -1 but got ` +
                `${d} for the slice() size at index ${i}.`);
            return x.shape[i] - begin_[i];
        }
    });
    return [begin_, size_];
}
// Convert the slicing specification from a sparse representation to a dense
// representation. This means that all ellipses and newaxis are expanded out.
export function sliceInfo(xShape, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask) {
    let stridesNonNull;
    if (strides == null) {
        stridesNonNull = new Array(begin.length);
        stridesNonNull.fill(1);
    }
    else {
        stridesNonNull = strides;
    }
    // Only one non-zero bit is allowed in ellipsisMask, which means ellipsisMask
    // is a power of 2. Use bit compares to ensure ellipsisMask is 0 or a power
    // of 2. When i is a power of 2, i & (i - 1) is always 0.
    // Also ref:
    // https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
    if (ellipsisMask != null && (ellipsisMask & (ellipsisMask - 1)) !== 0) {
        throw new Error('Multiple ellipses in slice is not allowed.');
    }
    // Step 1: Account for ellipsis and new axis.
    // Check for ellipsis and count how many non-newaxis there are after.
    let ellipsisSeen = false;
    const sparseSpec = {
        dims: stridesNonNull.length,
        numAddAxisAfterEllipsis: 0,
        begin: begin.slice(),
        end: end.slice(),
        strides: stridesNonNull.slice(),
        beginMask,
        endMask,
        ellipsisMask,
        newAxisMask,
        shrinkAxisMask
    };
    for (let i = 0; i < sparseSpec.dims; i++) {
        if (ellipsisSeen && ((1 << i) & newAxisMask) !== 0) {
            sparseSpec.numAddAxisAfterEllipsis++;
        }
        if ((1 << i) & ellipsisMask) {
            ellipsisSeen = true;
        }
    }
    // If no ellipsis insert one at the end.
    if (!ellipsisSeen) {
        sparseSpec.ellipsisMask |= (1 << sparseSpec.dims);
        sparseSpec.dims++; // this effects loop iteration below
    }
    // Step 2: Make a sparse spec into a full index spec.
    //
    // The sparse spec deos not correspond to the number of dimensions.
    // Make a dense spec that cooresponds to the number of dimensions.
    //
    // For example suppose foo[...,3:] on foo.shape = [2, 2, 3] then we need to
    // produce the missing beginMask for the first two dimensions i.e. from
    // beginMaskSpec = 0, endMaskSpec = 2, we achieve beginMask = 6 (110),
    // endMask = 7 (111).
    const denseSpec = {
        dims: xShape.length,
        beginMask: 0,
        endMask: 0,
        beginValid: false,
        endValid: false
    };
    buildDenseSpec(sparseSpec, denseSpec);
    // Step 3: Make implicit ranges (non-zero beginMasks and endMasks) explicit
    // and bounds check.
    let isIdentity = true;
    let sliceDim0 = true;
    let isSimpleSlice = true;
    const processingShape = [];
    const finalShape = [];
    for (let i = 0; i < xShape.length; ++i) {
        if (denseSpec.strides[i] === 0) {
            throw Error(`strides[${i}] must be non-zero`);
        }
        const shrinkI = !!(denseSpec.shrinkAxisMask & (1 << i));
        const dimI = xShape[i];
        if (dimI === -1) {
            processingShape.push(shrinkI ? 1 : -1);
            continue;
        }
        const masks = [denseSpec.beginMask & (1 << i), denseSpec.endMask & (1 << i)];
        const validRange = [
            denseSpec.strides[i] > 0 ? 0 : -1,
            denseSpec.strides[i] > 0 ? dimI : dimI - 1
        ];
        if (shrinkI && denseSpec.strides[i] <= 0) {
            throw Error('only stride 1 allowed on non-range indexing.');
        }
        isSimpleSlice = isSimpleSlice && (denseSpec.strides[i] === 1);
        const beginAndEndMasked = !!((denseSpec.beginMask & (1 << i)) && (denseSpec.endMask & (1 << i)));
        if (denseSpec.beginValid && denseSpec.endValid) {
            if (shrinkI) {
                // If we are shrinking, the end index is now possibly incorrect. In
                // particular foo[-1] produces sparseBegin = -1, sparseEnd = 0.
                // and canonical puts these to n-1 and 0, which implies a degenerate
                // interval. Fortunately, it is now safe to re-create end as begin + 1.
                const xFwd = denseSpec.begin[i] < 0 ? dimI + denseSpec.begin[i] :
                    denseSpec.begin[i];
                denseSpec.begin[i] = xFwd;
                denseSpec.end[i] = denseSpec.begin[i] + 1;
                if (xFwd < 0 || xFwd >= dimI) {
                    throw Error(`slice index ${denseSpec.begin[i]} of dimension ${i} out of bounds.`);
                }
            }
            else {
                denseSpec.begin[i] = canonical(denseSpec.begin[i], 0, denseSpec.strides[i], dimI, masks, validRange);
                denseSpec.end[i] = canonical(denseSpec.end[i], 1, denseSpec.strides[i], dimI, masks, validRange);
            }
            // Update optimization values
            const takeAllInDimension = denseSpec.strides[i] === 1 &&
                denseSpec.begin[i] === 0 && denseSpec.end[i] === dimI;
            isIdentity = isIdentity && takeAllInDimension;
            sliceDim0 = sliceDim0 &&
                ((i === 0 && denseSpec.strides[i] === 1) || takeAllInDimension);
        }
        else {
            isIdentity =
                isIdentity && ((denseSpec.strides[i] === 1) && beginAndEndMasked);
            sliceDim0 = sliceDim0 &&
                ((i === 0 && denseSpec.strides[i] === 1) || beginAndEndMasked);
        }
        // Compute the processing shape (the intermediate Eigen will produce)
        let intervalLength;
        let knownInterval = false;
        if (denseSpec.beginValid && denseSpec.endValid) {
            intervalLength = denseSpec.end[i] - denseSpec.begin[i];
            knownInterval = true;
        }
        else if (shrinkI) {
            // The dimension is still known as 1 for the processingShape, but will be
            // discarded for the final shape.
            intervalLength = 1;
            knownInterval = true;
        }
        else if (beginAndEndMasked) {
            // Even if we don't have values for begin or end, we do know that this
            // dimension covers the whole interval. If we have shape information for
            // this dimension, that tells us the interval length.
            if (dimI >= 0) {
                if (denseSpec.strides[i] < 0) {
                    intervalLength = -dimI;
                }
                else {
                    intervalLength = dimI;
                }
                knownInterval = true;
            }
        }
        if (knownInterval) {
            let sizeI;
            // Hold zero if the interval is degenerate, otherwise account for
            // remainder
            if (intervalLength === 0 ||
                ((intervalLength < 0) !== (denseSpec.strides[i] < 0))) {
                sizeI = 0;
            }
            else {
                sizeI = Math.trunc(intervalLength / denseSpec.strides[i]) +
                    (intervalLength % denseSpec.strides[i] !== 0 ? 1 : 0);
            }
            processingShape.push(sizeI);
        }
        else {
            processingShape.push(-1);
        }
    }
    // Step 4: Compute the final shape
    //
    // newAxis will increase dimension by 1 (with a one-size dimension)
    // slices like foo[3, ...] will reduce dimension by 1.
    // This cannot be done earlier, because it depends on Step 3.
    for (let denseDim = 0; denseDim < denseSpec.finalShapeGatherIndices.length; ++denseDim) {
        const gatherIndex = denseSpec.finalShapeGatherIndices[denseDim];
        if (gatherIndex >= 0) {
            finalShape.push(processingShape[gatherIndex]);
        }
        else if (gatherIndex === NEW_AXIS) {
            finalShape.push(1);
        }
    }
    const finalShapeSparse = finalShape.filter((dim, i) => denseSpec.finalShapeGatherIndices[i] !== NEW_AXIS);
    return {
        finalShapeSparse,
        finalShape,
        isIdentity,
        sliceDim0,
        isSimpleSlice,
        begin: denseSpec.begin,
        end: denseSpec.end,
        strides: denseSpec.strides
    };
}
function buildDenseSpec(sparse, dense) {
    dense.beginMask = 0;
    dense.endMask = 0;
    dense.shrinkAxisMask = 0;
    let fullIndex = 0;
    dense.beginValid = sparse.begin != null;
    dense.endValid = sparse.end != null;
    dense.begin = new Array(dense.dims);
    dense.end = new Array(dense.dims);
    dense.strides = new Array(dense.dims);
    dense.finalShapeGatherIndices = [];
    dense.finalShapeGatherIndicesSparse = [];
    dense.inputShapeGatherIndicesSparse = new Array(dense.dims);
    for (let i = 0; i < sparse.dims; i++) {
        if ((1 << i) & sparse.ellipsisMask) {
            // Only the bit that has ellipsis will fall in this condition.
            // Expand the ellipsis into the appropriate indices
            // Note: this only works because we guaranteed one ellipsis.
            const nextIndex = Math.min(dense.dims - (sparse.dims - i) + 1 + sparse.numAddAxisAfterEllipsis, dense.dims);
            for (; fullIndex < nextIndex; fullIndex++) {
                // newAxis aren't real axis so you have to skip.
                dense.begin[fullIndex] = 0;
                dense.end[fullIndex] = 0;
                dense.strides[fullIndex] = 1;
                dense.beginMask |= (1 << fullIndex);
                dense.endMask |= (1 << fullIndex);
                dense.finalShapeGatherIndices.push(fullIndex);
                dense.finalShapeGatherIndicesSparse.push(-1);
                dense.inputShapeGatherIndicesSparse[fullIndex] = i;
            }
        }
        else if ((1 << i) & sparse.newAxisMask) {
            // Only the bit that has newAxis will fall in this condition.
            dense.finalShapeGatherIndices.push(NEW_AXIS);
            dense.finalShapeGatherIndicesSparse.push(-1);
        }
        else {
            if (fullIndex === dense.begin.length) {
                throw Error(`Index out of range using input dim ${fullIndex}; input ` +
                    `has only ${dense.dims} dims, ${dense.begin.length}.`);
            }
            // Gather slicing spec into appropriate index.
            if (sparse.begin != null) {
                dense.begin[fullIndex] = sparse.begin[i];
            }
            if (sparse.end != null) {
                dense.end[fullIndex] = sparse.end[i];
            }
            dense.strides[fullIndex] = sparse.strides[i];
            if (sparse.beginMask & (1 << i)) {
                dense.beginMask |= (1 << fullIndex);
            }
            if (sparse.endMask & (1 << i)) {
                dense.endMask |= (1 << fullIndex);
            }
            // If shrink, record where to get the dimensionality from (i.e. newAxis)
            // creates a fake 1 size dimension. Also remember shrink axis (now in
            // dense form) so we can ignore dense.end below.
            if (sparse.shrinkAxisMask & (1 << i)) {
                dense.finalShapeGatherIndices.push(SHRINK_AXIS);
                dense.finalShapeGatherIndicesSparse.push(-1);
                dense.shrinkAxisMask |= (1 << fullIndex);
            }
            else {
                dense.finalShapeGatherIndices.push(fullIndex);
                // Remember that where in the sparse shape the dense dim comes from.
                dense.finalShapeGatherIndicesSparse.push(i);
            }
            dense.inputShapeGatherIndicesSparse[fullIndex] = i;
            fullIndex++;
        }
    }
}
function canonical(x, c, strideI, dimI, masks, validRange) {
    if (masks[c]) {
        return strideI > 0 ? validRange[c] : validRange[(c + 1) & 1];
    }
    else {
        const xFwd = x < 0 ? dimI + x : x; // make negative indices positive
        return xFwd < validRange[0] ? validRange[0] :
            xFwd > validRange[1] ? validRange[1] : xFwd;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2xpY2VfdXRpbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3NsaWNlX3V0aWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxLQUFLLElBQUksTUFBTSxTQUFTLENBQUM7QUFFaEMsTUFBTSxRQUFRLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUFDcEIsTUFBTSxXQUFXLEdBQUcsQ0FBQyxDQUFDLENBQUM7QUE2RHZCLE1BQU0sVUFBVSxpQkFBaUIsQ0FDN0IsS0FBaUIsRUFBRSxLQUFlLEVBQUUsSUFBYztJQUNwRCxNQUFNLFNBQVMsR0FBRyxLQUFLLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQztJQUNyQyxJQUFJLENBQUMsTUFBTSxDQUNQLFNBQVMsS0FBSyxLQUFLLENBQUMsTUFBTSxFQUMxQixHQUFHLEVBQUUsQ0FBQyxpQkFBaUIsU0FBUyxzQkFBc0IsS0FBSyxRQUFRO1FBQy9ELGdDQUFnQyxTQUFTLElBQUksQ0FBQyxDQUFDO0lBQ3ZELElBQUksQ0FBQyxNQUFNLENBQ1AsU0FBUyxLQUFLLElBQUksQ0FBQyxNQUFNLEVBQ3pCLEdBQUcsRUFBRSxDQUFDLGlCQUFpQixTQUFTLHFCQUFxQixJQUFJLFFBQVE7UUFDN0QsZ0NBQWdDLFNBQVMsSUFBSSxDQUFDLENBQUM7SUFFdkQsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFNBQVMsRUFBRSxFQUFFLENBQUMsRUFBRTtRQUNsQyxJQUFJLENBQUMsTUFBTSxDQUNQLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFDcEMsR0FBRyxFQUFFLENBQUMsaUJBQWlCLFNBQVMsWUFBWSxDQUFDLFlBQVksQ0FBQyxJQUFJO1lBQzFELElBQUksS0FBSyxDQUFD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UyxDQUFDLENBQUM7Z0JBQzlDLG9FQUFvRTtnQkFDcEUsS0FBSyxDQUFDLDZCQUE2QixDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUM3QztZQUNELEtBQUssQ0FBQyw2QkFBNkIsQ0FBQyxTQUFTLENBQUMsR0FBRyxDQUFDLENBQUM7WUFDbkQsU0FBUyxFQUFFLENBQUM7U0FDYjtLQUNGO0FBQ0gsQ0FBQztBQUVELFNBQVMsU0FBUyxDQUNkLENBQVMsRUFBRSxDQUFTLEVBQUUsT0FBZSxFQUFFLElBQVksRUFBRSxLQUFlLEVBQ3BFLFVBQW9CO0lBQ3RCLElBQUksS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFO1FBQ1osT0FBTyxPQUFPLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztLQUM5RDtTQUFNO1FBQ0wsTUFBTSxJQUFJLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUUsaUNBQWlDO1FBQ3JFLE9BQU8sSUFBSSxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDZixJQUFJLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQztLQUMzRTtBQUNILENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7IFRlbnNvckluZm8gfSBmcm9tICcuLi90ZW5zb3JfaW5mbyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5jb25zdCBORVdfQVhJUyA9IC0yO1xuY29uc3QgU0hSSU5LX0FYSVMgPSAtMTtcblxuLy8gU3BhcnNlIHNsaWNpbmcgc3BlY2lmaWNhdGlvblxuLy8gaWYgb25lIGRvZXMgZm9vWzM6NSwgLi4uLCAtM10sIHRoZSBiZWdpbiwgZW5kIGFuZCBzdHJpZGVzIHdpbGwgaGF2ZSBsZW5ndGhcbi8vIG9mIDMuXG5pbnRlcmZhY2UgU3RyaWRlZFNsaWNlU3BhcnNlU3BlYyB7XG4gIGRpbXM6IG51bWJlcjtcbiAgbnVtQWRkQXhpc0FmdGVyRWxsaXBzaXM6IG51bWJlcjtcbiAgYmVnaW46IG51bWJlcltdO1xuICBlbmQ6IG51bWJlcltdO1xuICBzdHJpZGVzOiBudW1iZXJbXTtcbiAgYmVnaW5NYXNrOiBudW1iZXI7XG4gIGVuZE1hc2s6IG51bWJlcjtcbiAgZWxsaXBzaXNNYXNrOiBudW1iZXI7XG4gIG5ld0F4aXNNYXNrOiBudW1iZXI7XG4gIHNocmlua0F4aXNNYXNrOiBudW1iZXI7XG59XG5cbi8vIERlbnNlIHNsaWNpbmcgc3BlY2lmaWNhdGlvblxuLy8gYWxsIGVsbGlwc2VzIGFuZCBuZXdheGlzIGFyZSBleHBhbmRlZCBvdXQuIFNvIGlmIGZvb1szOjUsIC4uLiwgLTNdIHdoZXJlIGZvb1xuLy8gaXMgMTAgZGltZW5zaW9uYWwsIGVhY2ggYXJyYXkgb2YgYmVnaW4sIGVuZCwgc3RyaWRlcyB3aWxsIGhhdmUgMTAgZW50cmllc1xuLy8gd2hlcmUgYXMgdGhlIHNwYXJzZSBjYW4gaGF2ZSBsZW5ndGggbGVzcyB0aGFuIHRoZSByYW5rIG9mIGZvby5cbmludGVyZmFjZSBTdHJpZGVkU2xpY2VEZW5zZVNwZWMge1xuICBkaW1zOiBudW1iZXI7XG4gIGJlZ2luTWFzaz86IG51bWJlcjtcbiAgZW5kTWFzaz86IG51bWJlcjtcbiAgYmVnaW5WYWxpZDogYm9vbGVhbjtcbiAgZW5kVmFsaWQ6IGJvb2xlYW47XG4gIGJlZ2luPzogbnVtYmVyW107XG4gIGVuZD86IG51bWJlcltdO1xuICBzdHJpZGVzPzogbnVtYmVyW107XG4gIC8vIFRoaXMgYXJyYXkgaGVscHMgY29uc3RydWN0IHRoZSBmaW5hbCBzaGFwZSBvZiB0aGUgc2xpY2UuXG4gIC8vIFRoZSBmaW5hbCB0ZW5zb3IgaXMgcmVkdWNlZCBpbiByYW5rIHdoZW5ldmVyIGEgc2luZ2xlIGluZGV4IGUuZy4gZm9vWzNdXG4gIC8vIGlzIGNhbGxlZCBmb3IuIFRoZSBmaW5hbCB0ZW5zb3IgaW5jcmVhc2VzIGluIHJhbmsgd2l0aCBuZXdBeGlzIGVudHJpZXMuXG4gIC8vIElmIGFuIGluZGV4IGluIHRoaXMgYXJyYXkgaXMgcG9zaXRpdmUsIHRoZSBzaXplIG9mIHRoZSBkaW1lbnNpb24gaXNcbiAgLy8gb2J0YWluZWQgZnJvbSBjYW5vbmljYWwgZW5kLWJlZ2luLiAgT3RoZXJ3aXNlLCBpZiBpdCBpcyBhIE5FV19BWElTLCBpdCB3aWxsXG4gIC8vIGJlIDEuIEEgc2hydW5rIGRpbWVuc2lvbiBpcyBza2lwcGVkLlxuICBmaW5hbFNoYXBlR2F0aGVySW5kaWNlcz86IG51bWJlcltdO1xuICAvLyBUaGlzIGFycmF5IGhhcyB0aGUgc2FtZSBzaXplIGFzIGZpbmFsU2hhcGVHYXRoZXJJbmRpY2VzLCBidXQgaXQgcmVtZW1iZXJzXG4gIC8vIHRoZSBzcGFyc2UgaW5kZXggdGhhdCBhIGRpbWVuc2lvbiBjb21lcyBmcm9tLCBpbnN0ZWFkIG9mIGRlbnNlIGluZGV4LlxuICAvLyBBIC0xIGluIHRoaXMgdmVjdG9yIG1lYW5zIHRoZSBpbmRleCBpcyBub3QgZnJvbSB0aGUgc3BhcnNlIGlucHV0LlxuICBmaW5hbFNoYXBlR2F0aGVySW5kaWNlc1NwYXJzZT86IG51bWJlcltdO1xuICBpbnB1dFNoYXBlR2F0aGVySW5kaWNlc1NwYXJzZT86IG51bWJlcltdO1xuICAvLyBUaGUgZGVuc2UgaW5kZXhlZCBzaHJpbmsgbWFzayBpcyB3aGljaCBwcm9jZXNzaW5nIGRpbWVuc2lvbnMgc2hvdWxkIGJlXG4gIC8vIHNocnVuay4gRm9yIGV4YW1wbGUsIGlmIGZvby5zaGFwZSA9IFsxMCwgMTAsIDEwLCAxMF0sIGZvb1szLCAuLi4sIDVdIGhhc1xuICAvLyBzcGFyc2VTaHJpbmtBeGlzTWFzayBvZiA1ICgwMTAxKSBhbmQgZGVuc2VTaHJpbmtBeGlzTWFzayBvZiA5ICgxMDAxKSxcbiAgLy8geWllbGRpbmcgYSBmaW5hbCBzaGFwZSBbMTAsIDEwXS5cbiAgc2hyaW5rQXhpc01hc2s/OiBudW1iZXI7XG59XG5cbmV4cG9ydCB0eXBlIFNsaWNlSW5mbyA9IHtcbiAgZmluYWxTaGFwZVNwYXJzZTogbnVtYmVyW10sXG4gIGZpbmFsU2hhcGU6IG51bWJlcltdLFxuICBpc0lkZW50aXR5OiBib29sZWFuLFxuICBzbGljZURpbTA6IGJvb2xlYW4sXG4gIGlzU2ltcGxlU2xpY2U6IGJvb2xlYW4sXG4gIGJlZ2luOiBudW1iZXJbXSxcbiAgZW5kOiBudW1iZXJbXSxcbiAgc3RyaWRlczogbnVtYmVyW11cbn07XG5cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnRQYXJhbXNWYWxpZChcbiAgICBpbnB1dDogVGVuc29ySW5mbywgYmVnaW46IG51bWJlcltdLCBzaXplOiBudW1iZXJbXSk6IHZvaWQge1xuICBjb25zdCBpbnB1dFJhbmsgPSBpbnB1dC5zaGFwZS5sZW5ndGg7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgaW5wdXRSYW5rID09PSBiZWdpbi5sZW5ndGgsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gc2xpY2Uke2lucHV0UmFua31EOiBMZW5ndGggb2YgYmVnaW4gJHtiZWdpbn0gbXVzdCBgICtcbiAgICAgICAgICBgbWF0Y2ggdGhlIHJhbmsgb2YgdGhlIGFycmF5ICgke2lucHV0UmFua30pLmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGlucHV0UmFuayA9PT0gc2l6ZS5sZW5ndGgsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gc2xpY2Uke2lucHV0UmFua31EOiBMZW5ndGggb2Ygc2l6ZSAke3NpemV9IG11c3QgYCArXG4gICAgICAgICAgYG1hdGNoIHRoZSByYW5rIG9mIHRoZSBhcnJheSAoJHtpbnB1dFJhbmt9KS5gKTtcblxuICBmb3IgKGxldCBpID0gMDsgaSA8IGlucHV0UmFuazsgKytpKSB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGJlZ2luW2ldICsgc2l6ZVtpXSA8PSBpbnB1dC5zaGFwZVtpXSxcbiAgICAgICAgKCkgPT4gYEVycm9yIGluIHNsaWNlJHtpbnB1dFJhbmt9RDogYmVnaW5bJHtpfV0gKyBzaXplWyR7aX1dIGAgK1xuICAgICAgICAgICAgYCgke2JlZ2luW2ldICsgc2l6ZVtpXX0pIHdvdWxkIG92ZXJmbG93IGlucHV0LnNoYXBlWyR7aX1dICgke1xuICAgICAgICAgICAgICAgICAgaW5wdXQuc2hhcGVbaV19KWApO1xuICB9XG59XG5cbi8qKiBDb252ZXJ0cyBhIGJpbmFyeSBtYXNrIHRvIGFuIGFycmF5IG9mIGF4ZXMuIFVzZWQgaW4gc3RyaWRlZFNsaWNlKCkuICovXG5leHBvcnQgZnVuY3Rpb24gbWFza1RvQXhlcyhtYXNrOiBudW1iZXIpOiBudW1iZXJbXSB7XG4gIGNvbnN0IGF4ZXMgPSBbXTtcbiAgbGV0IGF4aXMgPSAwO1xuICB3aGlsZSAobWFzayA+IDApIHtcbiAgICBpZiAobWFzayAmIDEpIHtcbiAgICAgIGF4ZXMucHVzaChheGlzKTtcbiAgICB9XG4gICAgbWFzayAvPSAyO1xuICAgIGF4aXMrKztcbiAgfVxuICByZXR1cm4gYXhlcztcbn1cblxuLyoqIENvbXB1dGVzIHRoZSBvdXRwdXQgc2hhcGUgZ2l2ZW4gdGhlIHN0cmlkZWQgc2xpY2UgcGFyYW1zLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVPdXRTaGFwZShcbiAgICBiZWdpbjogbnVtYmVyW10sIGVuZDogbnVtYmVyW10sIHN0cmlkZXM6IG51bWJlcltdKTogbnVtYmVyW10ge1xuICBjb25zdCBzaXplID0gW107XG4gIGZvciAobGV0IGF4aXMgPSAwOyBheGlzIDwgYmVnaW4ubGVuZ3RoOyBheGlzKyspIHtcbiAgICBzaXplW2F4aXNdID0gTWF0aC5jZWlsKChlbmRbYXhpc10gLSBiZWdpbltheGlzXSkgLyBzdHJpZGVzW2F4aXNdKTtcbiAgfVxuICByZXR1cm4gc2l6ZTtcbn1cblxuLy8gQ3JlYXRlcyBmdWxsIHNlbGVjdGlvbiBhdCB0aGUgZWxpZGVkIGRpbWVuc2lvbnMuIElmIHRoZSBkaW1lbnNpb24gbWF0Y2hlc1xuLy8gdGhlIGVsbGlwc2lzIG1hc2ssIG92ZXJyaWRlIHRoZSBjdXJyZW50IHN0cmlkZSB2YWx1ZS4gT3RoZXJ3aXNlLCBpbnNlcnQuXG5leHBvcnQgZnVuY3Rpb24gc3RyaWRlc1dpdGhFbGlkZWREaW1zKFxuICAgIHN0cmlkZXM6IG51bWJlcltdLCBlbGxpcHNpc0luc2VydGlvbkluZGV4OiBudW1iZXIsIG51bUVsaWRlZEF4ZXM6IG51bWJlcixcbiAgICBpbnB1dFNoYXBlOiBudW1iZXJbXSk6IG51bWJlcltdIHtcbiAgY29uc3QgbmV3U3RyaWRlcyA9IFsuLi5zdHJpZGVzXTtcbiAgZm9yIChsZXQgaSA9IG5ld1N0cmlkZXMubGVuZ3RoOyBpIDwgaW5wdXRTaGFwZS5sZW5ndGg7IGkrKykge1xuICAgIG5ld1N0cmlkZXMucHVzaCgxKTtcbiAgfVxuICBmb3IgKGxldCBpID0gMDsgaSA8IG51bUVsaWRlZEF4ZXM7IGkrKykge1xuICAgIGlmIChpID09PSAwKSB7XG4gICAgICBuZXdTdHJpZGVzW2VsbGlwc2lzSW5zZXJ0aW9uSW5kZXhdID0gMTtcbiAgICB9IGVsc2Uge1xuICAgICAgbmV3U3RyaWRlcy5zcGxpY2UoXG4gICAgICAgICAgZWxsaXBzaXNJbnNlcnRpb25JbmRleCwgMCAvKiBudW0gZWxlbWVudHMgdG8gZGVsZXRlICovLFxuICAgICAgICAgIDEgLyogZWxlbWVudCB0byBhZGQgKi8pO1xuICAgICAgbmV3U3RyaWRlcy5wb3AoKTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIG5ld1N0cmlkZXM7XG59XG5cbmZ1bmN0aW9uIHVubm9ybWFsaXplQXhpcyhcbiAgICBlbGxpcHNpc0luc2VydGlvbkluZGV4OiBudW1iZXIsIG51bUVsaWRlZEF4ZXM6IG51bWJlcixcbiAgICBub3JtYWxpemVkQXhpczogbnVtYmVyKTogbnVtYmVyIHtcbiAgaWYgKG5vcm1hbGl6ZWRBeGlzIDw9IGVsbGlwc2lzSW5zZXJ0aW9uSW5kZXgpIHtcbiAgICByZXR1cm4gbm9ybWFsaXplZEF4aXM7XG4gIH1cblxuICByZXR1cm4gbm9ybWFsaXplZEF4aXMgLSAobnVtRWxpZGVkQXhlcyAtIDEpO1xufVxuXG5mdW5jdGlvbiBnZXRFbGlkZWRBeGVzKG51bUVsaWRlZEF4ZXM6IG51bWJlciwgZWxsaXBzaXNJbnNlcnRpb25JbmRleDogbnVtYmVyKSB7XG4gIGNvbnN0IGVsaWRlZEF4ZXMgPSBbXTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBudW1FbGlkZWRBeGVzOyBpKyspIHtcbiAgICBlbGlkZWRBeGVzLnB1c2goZWxsaXBzaXNJbnNlcnRpb25JbmRleCArIGkpO1xuICB9XG4gIHJldHVybiBlbGlkZWRBeGVzO1xufVxuXG4vLyBOb3JtYWxpemUgdGhlIHN0YXJ0LCBlbmQgYW5kIHN0cmlkZXMuXG5leHBvcnQgZnVuY3Rpb24gZ2V0Tm9ybWFsaXplZEF4ZXMoXG4gICAgaW5wdXRTaGFwZTogbnVtYmVyW10sIGVsbGlwc2lzQXhlczogbnVtYmVyW10sIG51bUludGVycG9sYXRlZEF4ZXM6IG51bWJlcixcbiAgICBiZWdpbjogbnVtYmVyW10sIGVuZDogbnVtYmVyW10sIHN0cmlkZXM6IG51bWJlcltdLCBiZWdpbk1hc2s6IG51bWJlcixcbiAgICBlbmRNYXNrOiBudW1iZXIsXG4gICAgZWxsaXBzaXNNYXNrOiBudW1iZXIpOiB7YmVnaW46IG51bWJlcltdLCBlbmQ6IG51bWJlcltdLCBzdHJpZGVzOiBudW1iZXJbXX0ge1xuICBjb25zdCBpbnB1dFJhbmsgPSBpbnB1dFNoYXBlLmxlbmd0aDtcbiAgbGV0IG5vcm1hbGl6ZWRCZWdpbiA9IG5ldyBBcnJheShpbnB1dFJhbmspLFxuICAgICAgbm9ybWFsaXplZEVuZCA9IG5ldyBBcnJheShpbnB1dFJhbmspLFxuICAgICAgbm9ybWFsaXplZFN0cmlkZXMgPSBuZXcgQXJyYXkoaW5wdXRSYW5rKTtcbiAgaWYgKGVsbGlwc2lzQXhlcy5sZW5ndGggJiYgbnVtSW50ZXJwb2xhdGVkQXhlcyA+IDApIHtcbiAgICBjb25zdCBmdWxsSW5kZXggPSBlbGxpcHNpc0F4ZXNbMF07XG5cbiAgICAvLyBUaGUgZWxsaXBzaXMgYXBwbGllcyB0byB0aGUgbWFza2VkIGluZGV4IGFzIHdlbGwgYXMgYW55IGRpbWVuc2lvbnNcbiAgICAvLyB0aGF0IGFyZSBpbnRlcnBvbGF0ZWQuXG4gICAgY29uc3QgbnVtRWxpZGVkQXhlcyA9IG51bUludGVycG9sYXRlZEF4ZXMgKyAxO1xuICAgIG5vcm1hbGl6ZWRCZWdpbiA9IHN0YXJ0SW5kaWNlc1dpdGhFbGlkZWREaW1zKFxuICAgICAgICBiZWdpbk1hc2ssIGZ1bGxJbmRleCwgbnVtRWxpZGVkQXhlcywgYmVnaW4sIGlucHV0U2hhcGUpO1xuICAgIG5vcm1hbGl6ZWRFbmQgPSBzdG9wSW5kaWNlc1dpdGhFbGlkZWREaW1zKFxuICAgICAgICBlbmRNYXNrLCBmdWxsSW5kZXgsIG51bUVsaWRlZEF4ZXMsIGVuZCwgaW5wdXRTaGFwZSk7XG4gICAgbm9ybWFsaXplZFN0cmlkZXMgPVxuICAgICAgICBzdHJpZGVzV2l0aEVsaWRlZERpbXMoc3RyaWRlcywgZnVsbEluZGV4LCBudW1FbGlkZWRBeGVzLCBpbnB1dFNoYXBlKTtcbiAgfSBlbHNlIHtcbiAgICBmb3IgKGxldCBheGlzID0gMDsgYXhpcyA8IGlucHV0UmFuazsgYXhpcysrKSB7XG4gICAgICBub3JtYWxpemVkQmVnaW5bYXhpc10gPSBzdGFydEZvckF4aXMoXG4gICAgICAgICAgYmVnaW5NYXNrLCBiZWdpbiwgc3RyaWRlcywgaW5wdXRTaGFwZSwgYXhpcywgZWxsaXBzaXNNYXNrKTtcbiAgICAgIG5vcm1hbGl6ZWRFbmRbYXhpc10gPVxuICAgICAgICAgIHN0b3BGb3JBeGlzKGVuZE1hc2ssIGVuZCwgc3RyaWRlcywgaW5wdXRTaGFwZSwgYXhpcywgZWxsaXBzaXNNYXNrKTtcbiAgICAgIG5vcm1hbGl6ZWRTdHJpZGVzW2F4aXNdID0gc3RyaWRlc0ZvckF4aXMoc3RyaWRlcywgYXhpcywgZWxsaXBzaXNNYXNrKTtcbiAgICB9XG4gIH1cblxuICByZXR1cm4ge1xuICAgIGJlZ2luOiBub3JtYWxpemVkQmVnaW4sXG4gICAgZW5kOiBub3JtYWxpemVkRW5kLFxuICAgIHN0cmlkZXM6IG5vcm1hbGl6ZWRTdHJpZGVzXG4gIH07XG59XG5cbi8vIENyZWF0ZXMgZnVsbCBzZWxlY3Rpb24gYXQgdGhlIGVsaWRlZCBkaW1lbnNpb25zLiBJZiB0aGUgZGltZW5zaW9uIG1hdGNoZXNcbi8vIHRoZSBlbGxpcHNpcyBtYXNrLCBvdmVycmlkZSB0aGUgY3VycmVudCBzdGFydCB2YWx1ZS4gT3RoZXJ3aXNlLCBpbnNlcnQuXG5leHBvcnQgZnVuY3Rpb24gc3RhcnRJbmRpY2VzV2l0aEVsaWRlZERpbXMoXG4gICAgYmVnaW5NYXNrOiBudW1iZXIsIGVsbGlwc2lzSW5zZXJ0aW9uSW5kZXg6IG51bWJlciwgbnVtRWxpZGVkQXhlczogbnVtYmVyLFxuICAgIG9yaWdpbmFsQmVnaW46IG51bWJlcltdLCBpbnB1dFNoYXBlOiBudW1iZXJbXSk6IG51bWJlcltdIHtcbiAgY29uc3QgbmV3SW5kaWNlcyA9IFsuLi5pbnB1dFNoYXBlXTtcbiAgY29uc3QgZWxpZGVkQXhlcyA9IGdldEVsaWRlZEF4ZXMobnVtRWxpZGVkQXhlcywgZWxsaXBzaXNJbnNlcnRpb25JbmRleCk7XG5cbiAgZm9yIChsZXQgYXhpcyA9IDA7IGF4aXMgPCBuZXdJbmRpY2VzLmxlbmd0aDsgYXhpcysrKSB7XG4gICAgaWYgKGVsaWRlZEF4ZXMuaW5kZXhPZihheGlzKSA+IC0xKSB7XG4gICAgICBuZXdJbmRpY2VzW2F4aXNdID0gMDtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3Qgb3JpZ2luYWxBeGlzID1cbiAgICAgICAgICB1bm5vcm1hbGl6ZUF4aXMoZWxsaXBzaXNJbnNlcnRpb25JbmRleCwgbnVtRWxpZGVkQXhlcywgYXhpcyk7XG4gICAgICBsZXQgb3JpZ2luYWxWYWx1ZSA9IG9yaWdpbmFsQmVnaW5bb3JpZ2luYWxBeGlzXTtcbiAgICAgIGlmIChiZWdpbk1hc2sgJiAxIDw8IG9yaWdpbmFsQXhpcykge1xuICAgICAgICBvcmlnaW5hbFZhbHVlID0gMDtcbiAgICAgIH1cblxuICAgICAgbmV3SW5kaWNlc1theGlzXSA9IG9yaWdpbmFsVmFsdWU7XG4gICAgfVxuICB9XG4gIHJldHVybiBuZXdJbmRpY2VzO1xufVxuXG4vLyBDcmVhdGVzIGZ1bGwgc2VsZWN0aW9uIGF0IHRoZSBlbGlkZWQgZGltZW5zaW9ucy4gSWYgdGhlIGRpbWVuc2lvbiBtYXRjaGVzXG4vLyB0aGUgZWxsaXBzaXMgbWFzaywgb3ZlcnJpZGUgdGhlIGN1cnJlbnQgc3RvcCB2YWx1ZS4gT3RoZXJ3aXNlLCBpbnNlcnQuXG5leHBvcnQgZnVuY3Rpb24gc3RvcEluZGljZXNXaXRoRWxpZGVkRGltcyhcbiAgICBlbmRNYXNrOiBudW1iZXIsIGVsbGlwc2lzSW5zZXJ0aW9uSW5kZXg6IG51bWJlciwgbnVtRWxpZGVkQXhlczogbnVtYmVyLFxuICAgIG9yaWdpbmFsRW5kOiBudW1iZXJbXSwgaW5wdXRTaGFwZTogbnVtYmVyW10pOiBudW1iZXJbXSB7XG4gIGNvbnN0IG5ld0luZGljZXMgPSBbLi4uaW5wdXRTaGFwZV07XG4gIGNvbnN0IGVsaWRlZEF4ZXMgPSBnZXRFbGlkZWRBeGVzKG51bUVsaWRlZEF4ZXMsIGVsbGlwc2lzSW5zZXJ0aW9uSW5kZXgpO1xuXG4gIGZvciAobGV0IGF4aXMgPSAwOyBheGlzIDwgbmV3SW5kaWNlcy5sZW5ndGg7IGF4aXMrKykge1xuICAgIGlmIChlbGlkZWRBeGVzLmluZGV4T2YoYXhpcykgPiAtMSkge1xuICAgICAgbmV3SW5kaWNlc1theGlzXSA9IE51bWJlci5NQVhfU0FGRV9JTlRFR0VSO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBvcmlnaW5hbEF4aXMgPVxuICAgICAgICAgIHVubm9ybWFsaXplQXhpcyhlbGxpcHNpc0luc2VydGlvbkluZGV4LCBudW1FbGlkZWRBeGVzLCBheGlzKTtcbiAgICAgIGxldCBvcmlnaW5hbFZhbHVlID0gb3JpZ2luYWxFbmRbb3JpZ2luYWxBeGlzXTtcbiAgICAgIGlmIChlbmRNYXNrICYgMSA8PCBvcmlnaW5hbEF4aXMpIHtcbiAgICAgICAgb3JpZ2luYWxWYWx1ZSA9IE51bWJlci5NQVhfU0FGRV9JTlRFR0VSO1xuICAgICAgfVxuICAgICAgbmV3SW5kaWNlc1theGlzXSA9IG9yaWdpbmFsVmFsdWU7XG4gICAgfVxuICB9XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuZXdJbmRpY2VzLmxlbmd0aDsgaSsrKSB7XG4gICAgLy8gSGFuZGxlIG5lZ2F0aXZlIGluZGljZXNcbiAgICBjb25zdCBheGlzU2l6ZSA9IGlucHV0U2hhcGVbaV07XG4gICAgaWYgKG5ld0luZGljZXNbaV0gPCAwKSB7XG4gICAgICBuZXdJbmRpY2VzW2ldICs9IGF4aXNTaXplO1xuICAgIH1cbiAgICBuZXdJbmRpY2VzW2ldID0gdXRpbC5jbGFtcCgwLCBuZXdJbmRpY2VzW2ldLCBpbnB1dFNoYXBlW2ldKTtcbiAgfVxuICByZXR1cm4gbmV3SW5kaWNlcztcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHN0cmlkZXNGb3JBeGlzKFxuICAgIHN0cmlkZXM6IG51bWJlcltdLCBheGlzOiBudW1iZXIsIGVsbGlwc2lzTWFzazogbnVtYmVyKTogbnVtYmVyIHtcbiAgbGV0IHN0cmlkZSA9IHN0cmlkZXNbYXhpc107XG4gIGlmIChlbGxpcHNpc01hc2sgJiAoMSA8PCBheGlzKSB8fCBzdHJpZGUgPT0gbnVsbCkge1xuICAgIHN0cmlkZSA9IDE7XG4gIH1cblxuICByZXR1cm4gc3RyaWRlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gc3RhcnRGb3JBeGlzKFxuICAgIGJlZ2luTWFzazogbnVtYmVyLCBzdGFydEluZGljZXM6IG51bWJlcltdLCBzdHJpZGVzOiBudW1iZXJbXSxcbiAgICBpbnB1dFNoYXBlOiBudW1iZXJbXSwgYXhpczogbnVtYmVyLCBlbGxpcHNpc01hc2s6IG51bWJlcik6IG51bWJlciB7XG4gIC8vIEJlZ2luIHdpdGggdGhlIHNwZWNpZmllZCBpbmRleFxuICBsZXQgc3RhcnQgPSBzdGFydEluZGljZXNbYXhpc107XG4gIGNvbnN0IHN0cmlkZSA9IHN0cmlkZXNbYXhpc10gfHwgMTtcblxuICAvLyBDaGVjayB0aGUgYXhpcyBiaXQgZnJvbSByaWdodCBvZiBtYXNrZWQgYXhlcywgb3IgdGhlIGJlZ2luIGluZGV4IGlzIG5vdCBzZXRcbiAgLy8gZm9yIHRoZSBheGlzLlxuICBpZiAoYmVnaW5NYXNrICYgMSA8PCBheGlzIHx8IGVsbGlwc2lzTWFzayAmIDEgPDwgYXhpcyB8fCBzdGFydCA9PSBudWxsKSB7XG4gICAgaWYgKHN0cmlkZSA+IDApIHtcbiAgICAgIC8vIEZvcndhcmQgaXRlcmF0aW9uIC0gdXNlIHRoZSBmaXJzdCBlbGVtZW50LiBUaGVzZSB2YWx1ZXMgd2lsbCBnZXRcbiAgICAgIC8vIGNsYW1wZWQgYmVsb3cgKE5vdGU6IFdlIGNvdWxkIGhhdmUgc2V0IHRoZW0gdG8gMCBhbmQgYXhpc19zaXplLTEsIGJ1dFxuICAgICAgLy8gdXNlIGxvd2VzdCgpIGFuZCBtYXgoKSB0byBtYWludGFpbiBzeW1tZXRyeSB3aXRoIFN0b3BGb3JBeGlzKCkpXG4gICAgICBzdGFydCA9IE51bWJlci5NSU5fU0FGRV9JTlRFR0VSO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBCYWNrd2FyZCBpdGVyYXRpb24gLSB1c2UgdGhlIGxhc3QgZWxlbWVudC5cbiAgICAgIHN0YXJ0ID0gTnVtYmVyLk1BWF9TQUZFX0lOVEVHRVI7XG4gICAgfVxuICB9XG5cbiAgLy8gSGFuZGxlIG5lZ2F0aXZlIGluZGljZXNcbiAgY29uc3QgYXhpc1NpemUgPSBpbnB1dFNoYXBlW2F4aXNdO1xuICBpZiAoc3RhcnQgPCAwKSB7XG4gICAgc3RhcnQgKz0gYXhpc1NpemU7XG4gIH1cblxuICAvLyBDbGFtcGluZ1xuICBzdGFydCA9IHV0aWwuY2xhbXAoMCwgc3RhcnQsIGF4aXNTaXplIC0gMSk7XG5cbiAgcmV0dXJuIHN0YXJ0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gc3RvcEZvckF4aXMoXG4gICAgZW5kTWFzazogbnVtYmVyLCBzdG9wSW5kaWNlczogbnVtYmVyW10sIHN0cmlkZXM6IG51bWJlcltdLFxuICAgIGlucHV0U2hhcGU6IG51bWJlcltdLCBheGlzOiBudW1iZXIsIGVsbGlwc2lzTWFzazogbnVtYmVyKTogbnVtYmVyIHtcbiAgLy8gQmVnaW4gd2l0aCB0aGUgc3BlY2lmaWVkIGluZGV4XG4gIGxldCBzdG9wID0gc3RvcEluZGljZXNbYXhpc107XG4gIGNvbnN0IHN0cmlkZSA9IHN0cmlkZXNbYXhpc10gfHwgMTtcblxuICAvLyBDaGVjayB0aGUgYXhpcyBiaXQgZnJvbSByaWdodCBvZiBtYXNrZWQgYXhlcywgb3IgaWYgdGhlIHN0b3AgaW5kZXggaXMgbm90XG4gIC8vIHNldCBmb3IgdGhpcyBheGlzLlxuICBpZiAoZW5kTWFzayAmICgxIDw8IGF4aXMpIHx8IGVsbGlwc2lzTWFzayAmICgxIDw8IGF4aXMpIHx8IHN0b3AgPT0gbnVsbCkge1xuICAgIGlmIChzdHJpZGUgPiAwKSB7XG4gICAgICAvLyBGb3J3YXJkIGl0ZXJhdGlvbiAtIHVzZSB0aGUgbGFzdCBlbGVtZW50LiBUaGVzZSB2YWx1ZXMgd2lsbCBnZXRcbiAgICAgIC8vIGNsYW1wZWQgYmVsb3dcbiAgICAgIHN0b3AgPSBOdW1iZXIuTUFYX1NBRkVfSU5URUdFUjtcbiAgICB9IGVsc2Uge1xuICAgICAgLy8gQmFja3dhcmQgaXRlcmF0aW9uIC0gdXNlIHRoZSBmaXJzdCBlbGVtZW50LlxuICAgICAgc3RvcCA9IE51bWJlci5NSU5fU0FGRV9JTlRFR0VSO1xuICAgIH1cbiAgfVxuXG4gIC8vIEhhbmRsZSBuZWdhdGl2ZSBpbmRpY2VzXG4gIGNvbnN0IGF4aXNTaXplID0gaW5wdXRTaGFwZVtheGlzXTtcbiAgaWYgKHN0b3AgPCAwKSB7XG4gICAgc3RvcCArPSBheGlzU2l6ZTtcbiAgfVxuXG4gIC8vIENsYW1waW5nXG4gIC8vIEJlY2F1c2UgdGhlIGVuZCBpbmRleCBwb2ludHMgb25lIHBhc3QgdGhlIGxhc3QgZWxlbWVudCwgd2UgbmVlZCBzbGlnaHRseVxuICAvLyBkaWZmZXJlbnQgY2xhbXBpbmcgcmFuZ2VzIGRlcGVuZGluZyBvbiB0aGUgZGlyZWN0aW9uLlxuICBpZiAoc3RyaWRlID4gMCkge1xuICAgIC8vIEZvcndhcmQgaXRlcmF0aW9uXG4gICAgc3RvcCA9IHV0aWwuY2xhbXAoMCwgc3RvcCwgYXhpc1NpemUpO1xuICB9IGVsc2Uge1xuICAgIC8vIEJhY2t3YXJkIGl0ZXJhdGlvblxuICAgIHN0b3AgPSB1dGlsLmNsYW1wKC0xLCBzdG9wLCBheGlzU2l6ZSAtIDEpO1xuICB9XG5cbiAgcmV0dXJuIHN0b3A7XG59XG5cbi8qKlxuICogUmV0dXJucyB0cnVlIGlmIHRoZSBzbGljZSBvY2N1cGllcyBhIGNvbnRpbm91cyBzZXQgb2YgZWxlbWVudHMgaW4gdGhlXG4gKiAnZmxhdCcgc3BhY2UuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBpc1NsaWNlQ29udGlub3VzKFxuICAgIHNoYXBlOiBudW1iZXJbXSwgYmVnaW46IG51bWJlcltdLCBzaXplOiBudW1iZXJbXSkge1xuICAvLyBJbmRleCBvZiB0aGUgZmlyc3QgYXhpcyB0aGF0IGhhcyBzaXplID4gMS5cbiAgbGV0IGZpcnN0Tm9uT25lQXhpcyA9IHNpemUubGVuZ3RoO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IHNpemUubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAoc2l6ZVtpXSA+IDEpIHtcbiAgICAgIGZpcnN0Tm9uT25lQXhpcyA9IGk7XG4gICAgICBicmVhaztcbiAgICB9XG4gIH1cblxuICBmb3IgKGxldCBpID0gZmlyc3ROb25PbmVBeGlzICsgMTsgaSA8IHNpemUubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAoYmVnaW5baV0gPiAwIHx8IHNpemVbaV0gIT09IHNoYXBlW2ldKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICB9XG4gIHJldHVybiB0cnVlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZUZsYXRPZmZzZXQoYmVnaW46IG51bWJlcltdLCBzdHJpZGVzOiBudW1iZXJbXSk6IG51bWJlciB7XG4gIGxldCBmbGF0T2Zmc2V0ID0gYmVnaW4ubGVuZ3RoID4gMCA/IGJlZ2luW2JlZ2luLmxlbmd0aCAtIDFdIDogMTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBiZWdpbi5sZW5ndGggLSAxOyBpKyspIHtcbiAgICBmbGF0T2Zmc2V0ICs9IGJlZ2luW2ldICogc3RyaWRlc1tpXTtcbiAgfVxuICByZXR1cm4gZmxhdE9mZnNldDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHBhcnNlU2xpY2VQYXJhbXMoXG4gICAgeDogVGVuc29ySW5mbywgYmVnaW46IG51bWJlcnxudW1iZXJbXSwgc2l6ZT86IG51bWJlcnxudW1iZXJbXSkge1xuICAvLyBUaGUgZm9sbG93aW5nIGxvZ2ljIGFsbG93cyBmb3IgbW9yZSBlcmdvbm9taWMgY2FsbHMuXG4gIGxldCBiZWdpbl86IG51bWJlcltdO1xuICBjb25zdCB4UmFuayA9IHguc2hhcGUubGVuZ3RoO1xuICBpZiAodHlwZW9mIGJlZ2luID09PSAnbnVtYmVyJykge1xuICAgIGJlZ2luXyA9IFtiZWdpbiwgLi4ubmV3IEFycmF5KHhSYW5rIC0gMSkuZmlsbCgwKV07XG4gIH0gZWxzZSBpZiAoYmVnaW4ubGVuZ3RoIDwgeFJhbmspIHtcbiAgICBiZWdpbl8gPSBiZWdpbi5jb25jYXQobmV3IEFycmF5KHhSYW5rIC0gYmVnaW4ubGVuZ3RoKS5maWxsKDApKTtcbiAgfSBlbHNlIHtcbiAgICBiZWdpbl8gPSBiZWdpbi5zbGljZSgpO1xuICB9XG4gIGJlZ2luXy5mb3JFYWNoKGQgPT4ge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBkICE9PSAtMSwgKCkgPT4gJ3NsaWNlKCkgZG9lcyBub3Qgc3VwcG9ydCBuZWdhdGl2ZSBiZWdpbiBpbmRleGluZy4nKTtcbiAgfSk7XG4gIGxldCBzaXplXzogbnVtYmVyW107XG4gIGlmIChzaXplID09IG51bGwpIHtcbiAgICBzaXplXyA9IG5ldyBBcnJheSh4UmFuaykuZmlsbCgtMSk7XG4gIH0gZWxzZSBpZiAodHlwZW9mIHNpemUgPT09ICdudW1iZXInKSB7XG4gICAgc2l6ZV8gPSBbc2l6ZSwgLi4ubmV3IEFycmF5KHhSYW5rIC0gMSkuZmlsbCgtMSldO1xuICB9IGVsc2UgaWYgKHNpemUubGVuZ3RoIDwgeFJhbmspIHtcbiAgICBzaXplXyA9IHNpemUuY29uY2F0KG5ldyBBcnJheSh4UmFuayAtIHNpemUubGVuZ3RoKS5maWxsKC0xKSk7XG4gIH0gZWxzZSB7XG4gICAgc2l6ZV8gPSBzaXplO1xuICB9XG4gIHNpemVfID0gc2l6ZV8ubWFwKChkLCBpKSA9PiB7XG4gICAgaWYgKGQgPj0gMCkge1xuICAgICAgcmV0dXJuIGQ7XG4gICAgfSBlbHNlIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGQgPT09IC0xLFxuICAgICAgICAgICgpID0+IGBOZWdhdGl2ZSBzaXplIHZhbHVlcyBzaG91bGQgYmUgZXhhY3RseSAtMSBidXQgZ290IGAgK1xuICAgICAgICAgICAgICBgJHtkfSBmb3IgdGhlIHNsaWNlKCkgc2l6ZSBhdCBpbmRleCAke2l9LmApO1xuICAgICAgcmV0dXJuIHguc2hhcGVbaV0gLSBiZWdpbl9baV07XG4gICAgfVxuICB9KTtcbiAgcmV0dXJuIFtiZWdpbl8sIHNpemVfXTtcbn1cblxuLy8gQ29udmVydCB0aGUgc2xpY2luZyBzcGVjaWZpY2F0aW9uIGZyb20gYSBzcGFyc2UgcmVwcmVzZW50YXRpb24gdG8gYSBkZW5zZVxuLy8gcmVwcmVzZW50YXRpb24uIFRoaXMgbWVhbnMgdGhhdCBhbGwgZWxsaXBzZXMgYW5kIG5ld2F4aXMgYXJlIGV4cGFuZGVkIG91dC5cbmV4cG9ydCBmdW5jdGlvbiBzbGljZUluZm8oXG4gICAgeFNoYXBlOiBudW1iZXJbXSwgYmVnaW46IG51bWJlcltdLCBlbmQ6IG51bWJlcltdLCBzdHJpZGVzOiBudW1iZXJbXSxcbiAgICBiZWdpbk1hc2s6IG51bWJlciwgZW5kTWFzazogbnVtYmVyLCBlbGxpcHNpc01hc2s6IG51bWJlcixcbiAgICBuZXdBeGlzTWFzazogbnVtYmVyLCBzaHJpbmtBeGlzTWFzazogbnVtYmVyKTogU2xpY2VJbmZvIHtcbiAgbGV0IHN0cmlkZXNOb25OdWxsO1xuICBpZiAoc3RyaWRlcyA9PSBudWxsKSB7XG4gICAgc3RyaWRlc05vbk51bGwgPSBuZXcgQXJyYXkoYmVnaW4ubGVuZ3RoKTtcbiAgICBzdHJpZGVzTm9uTnVsbC5maWxsKDEpO1xuICB9IGVsc2Uge1xuICAgIHN0cmlkZXNOb25OdWxsID0gc3RyaWRlcztcbiAgfVxuXG4gIC8vIE9ubHkgb25lIG5vbi16ZXJvIGJpdCBpcyBhbGxvd2VkIGluIGVsbGlwc2lzTWFzaywgd2hpY2ggbWVhbnMgZWxsaXBzaXNNYXNrXG4gIC8vIGlzIGEgcG93ZXIgb2YgMi4gVXNlIGJpdCBjb21wYXJlcyB0byBlbnN1cmUgZWxsaXBzaXNNYXNrIGlzIDAgb3IgYSBwb3dlclxuICAvLyBvZiAyLiBXaGVuIGkgaXMgYSBwb3dlciBvZiAyLCBpICYgKGkgLSAxKSBpcyBhbHdheXMgMC5cbiAgLy8gQWxzbyByZWY6XG4gIC8vIGh0dHBzOi8vc3RhY2tvdmVyZmxvdy5jb20vcXVlc3Rpb25zLzYwMDI5My9ob3ctdG8tY2hlY2staWYtYS1udW1iZXItaXMtYS1wb3dlci1vZi0yXG4gIGlmIChlbGxpcHNpc01hc2sgIT0gbnVsbCAmJiAoZWxsaXBzaXNNYXNrICYgKGVsbGlwc2lzTWFzayAtIDEpKSAhPT0gMCkge1xuICAgIHRocm93IG5ldyBFcnJvcignTXVsdGlwbGUgZWxsaXBzZXMgaW4gc2xpY2UgaXMgbm90IGFsbG93ZWQuJyk7XG4gIH1cblxuICAvLyBTdGVwIDE6IEFjY291bnQgZm9yIGVsbGlwc2lzIGFuZCBuZXcgYXhpcy5cbiAgLy8gQ2hlY2sgZm9yIGVsbGlwc2lzIGFuZCBjb3VudCBob3cgbWFueSBub24tbmV3YXhpcyB0aGVyZSBhcmUgYWZ0ZXIuXG4gIGxldCBlbGxpcHNpc1NlZW4gPSBmYWxzZTtcblxuICBjb25zdCBzcGFyc2VTcGVjOiBTdHJpZGVkU2xpY2VTcGFyc2VTcGVjID0ge1xuICAgIGRpbXM6IHN0cmlkZXNOb25OdWxsLmxlbmd0aCxcbiAgICBudW1BZGRBeGlzQWZ0ZXJFbGxpcHNpczogMCxcbiAgICBiZWdpbjogYmVnaW4uc2xpY2UoKSxcbiAgICBlbmQ6IGVuZC5zbGljZSgpLFxuICAgIHN0cmlkZXM6IHN0cmlkZXNOb25OdWxsLnNsaWNlKCksXG4gICAgYmVnaW5NYXNrLFxuICAgIGVuZE1hc2ssXG4gICAgZWxsaXBzaXNNYXNrLFxuICAgIG5ld0F4aXNNYXNrLFxuICAgIHNocmlua0F4aXNNYXNrXG4gIH07XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBzcGFyc2VTcGVjLmRpbXM7IGkrKykge1xuICAgIGlmIChlbGxpcHNpc1NlZW4gJiYgKCgxIDw8IGkpICYgbmV3QXhpc01hc2spICE9PSAwKSB7XG4gICAgICBzcGFyc2VTcGVjLm51bUFkZEF4aXNBZnRlckVsbGlwc2lzKys7XG4gICAgfVxuICAgIGlmICgoMSA8PCBpKSAmIGVsbGlwc2lzTWFzaykge1xuICAgICAgZWxsaXBzaXNTZWVuID0gdHJ1ZTtcbiAgICB9XG4gIH1cbiAgLy8gSWYgbm8gZWxsaXBzaXMgaW5zZXJ0IG9uZSBhdCB0aGUgZW5kLlxuICBpZiAoIWVsbGlwc2lzU2Vlbikge1xuICAgIHNwYXJzZVNwZWMuZWxsaXBzaXNNYXNrIHw9ICgxIDw8IHNwYXJzZVNwZWMuZGltcyk7XG4gICAgc3BhcnNlU3BlYy5kaW1zKys7ICAvLyB0aGlzIGVmZmVjdHMgbG9vcCBpdGVyYXRpb24gYmVsb3dcbiAgfVxuXG4gIC8vIFN0ZXAgMjogTWFrZSBhIHNwYXJzZSBzcGVjIGludG8gYSBmdWxsIGluZGV4IHNwZWMuXG4gIC8vXG4gIC8vIFRoZSBzcGFyc2Ugc3BlYyBkZW9zIG5vdCBjb3JyZXNwb25kIHRvIHRoZSBudW1iZXIgb2YgZGltZW5zaW9ucy5cbiAgLy8gTWFrZSBhIGRlbnNlIHNwZWMgdGhhdCBjb29yZXNwb25kcyB0byB0aGUgbnVtYmVyIG9mIGRpbWVuc2lvbnMuXG4gIC8vXG4gIC8vIEZvciBleGFtcGxlIHN1cHBvc2UgZm9vWy4uLiwzOl0gb24gZm9vLnNoYXBlID0gWzIsIDIsIDNdIHRoZW4gd2UgbmVlZCB0b1xuICAvLyBwcm9kdWNlIHRoZSBtaXNzaW5nIGJlZ2luTWFzayBmb3IgdGhlIGZpcnN0IHR3byBkaW1lbnNpb25zIGkuZS4gZnJvbVxuICAvLyBiZWdpbk1hc2tTcGVjID0gMCwgZW5kTWFza1NwZWMgPSAyLCB3ZSBhY2hpZXZlIGJlZ2luTWFzayA9IDYgKDExMCksXG4gIC8vIGVuZE1hc2sgPSA3ICgxMTEpLlxuICBjb25zdCBkZW5zZVNwZWM6IFN0cmlkZWRTbGljZURlbnNlU3BlYyA9IHtcbiAgICBkaW1zOiB4U2hhcGUubGVuZ3RoLFxuICAgIGJlZ2luTWFzazogMCxcbiAgICBlbmRNYXNrOiAwLFxuICAgIGJlZ2luVmFsaWQ6IGZhbHNlLFxuICAgIGVuZFZhbGlkOiBmYWxzZVxuICB9O1xuXG4gIGJ1aWxkRGVuc2VTcGVjKHNwYXJzZVNwZWMsIGRlbnNlU3BlYyk7XG5cbiAgLy8gU3RlcCAzOiBNYWtlIGltcGxpY2l0IHJhbmdlcyAobm9uLXplcm8gYmVnaW5NYXNrcyBhbmQgZW5kTWFza3MpIGV4cGxpY2l0XG4gIC8vIGFuZCBib3VuZHMgY2hlY2suXG4gIGxldCBpc0lkZW50aXR5ID0gdHJ1ZTtcbiAgbGV0IHNsaWNlRGltMCA9IHRydWU7XG4gIGxldCBpc1NpbXBsZVNsaWNlID0gdHJ1ZTtcbiAgY29uc3QgcHJvY2Vzc2luZ1NoYXBlID0gW107XG4gIGNvbnN0IGZpbmFsU2hhcGUgPSBbXTtcblxuICBmb3IgKGxldCBpID0gMDsgaSA8IHhTaGFwZS5sZW5ndGg7ICsraSkge1xuICAgIGlmIChkZW5zZVNwZWMuc3RyaWRlc1tpXSA9PT0gMCkge1xuICAgICAgdGhyb3cgRXJyb3IoYHN0cmlkZXNbJHtpfV0gbXVzdCBiZSBub24temVyb2ApO1xuICAgIH1cbiAgICBjb25zdCBzaHJpbmtJID0gISEoZGVuc2VTcGVjLnNocmlua0F4aXNNYXNrICYgKDEgPDwgaSkpO1xuICAgIGNvbnN0IGRpbUkgPSB4U2hhcGVbaV07XG4gICAgaWYgKGRpbUkgPT09IC0xKSB7XG4gICAgICBwcm9jZXNzaW5nU2hhcGUucHVzaChzaHJpbmtJID8gMSA6IC0xKTtcbiAgICAgIGNvbnRpbnVlO1xuICAgIH1cblxuICAgIGNvbnN0IG1hc2tzID1cbiAgICAgICAgW2RlbnNlU3BlYy5iZWdpbk1hc2sgJiAoMSA8PCBpKSwgZGVuc2VTcGVjLmVuZE1hc2sgJiAoMSA8PCBpKV07XG4gICAgY29uc3QgdmFsaWRSYW5nZSA9IFtcbiAgICAgIGRlbnNlU3BlYy5zdHJpZGVzW2ldID4gMCA/IDAgOiAtMSxcbiAgICAgIGRlbnNlU3BlYy5zdHJpZGVzW2ldID4gMCA/IGRpbUkgOiBkaW1JIC0gMVxuICAgIF07XG5cbiAgICBpZiAoc2hyaW5rSSAmJiBkZW5zZVNwZWMuc3RyaWRlc1tpXSA8PSAwKSB7XG4gICAgICB0aHJvdyBFcnJvcignb25seSBzdHJpZGUgMSBhbGxvd2VkIG9uIG5vbi1yYW5nZSBpbmRleGluZy4nKTtcbiAgICB9XG5cbiAgICBpc1NpbXBsZVNsaWNlID0gaXNTaW1wbGVTbGljZSAmJiAoZGVuc2VTcGVjLnN0cmlkZXNbaV0gPT09IDEpO1xuXG4gICAgY29uc3QgYmVnaW5BbmRFbmRNYXNrZWQgPVxuICAgICAgICAhISgoZGVuc2VTcGVjLmJlZ2luTWFzayAmICgxIDw8IGkpKSAmJiAoZGVuc2VTcGVjLmVuZE1hc2sgJiAoMSA8PCBpKSkpO1xuXG4gICAgaWYgKGRlbnNlU3BlYy5iZWdpblZhbGlkICYmIGRlbnNlU3BlYy5lbmRWYWxpZCkge1xuICAgICAgaWYgKHNocmlua0kpIHtcbiAgICAgICAgLy8gSWYgd2UgYXJlIHNocmlua2luZywgdGhlIGVuZCBpbmRleCBpcyBub3cgcG9zc2libHkgaW5jb3JyZWN0LiBJblxuICAgICAgICAvLyBwYXJ0aWN1bGFyIGZvb1stMV0gcHJvZHVjZXMgc3BhcnNlQmVnaW4gPSAtMSwgc3BhcnNlRW5kID0gMC5cbiAgICAgICAgLy8gYW5kIGNhbm9uaWNhbCBwdXRzIHRoZXNlIHRvIG4tMSBhbmQgMCwgd2hpY2ggaW1wbGllcyBhIGRlZ2VuZXJhdGVcbiAgICAgICAgLy8gaW50ZXJ2YWwuIEZvcnR1bmF0ZWx5LCBpdCBpcyBub3cgc2FmZSB0byByZS1jcmVhdGUgZW5kIGFzIGJlZ2luICsgMS5cbiAgICAgICAgY29uc3QgeEZ3ZCA9IGRlbnNlU3BlYy5iZWdpbltpXSA8IDAgPyBkaW1JICsgZGVuc2VTcGVjLmJlZ2luW2ldIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBkZW5zZVNwZWMuYmVnaW5baV07XG4gICAgICAgIGRlbnNlU3BlYy5iZWdpbltpXSA9IHhGd2Q7XG4gICAgICAgIGRlbnNlU3BlYy5lbmRbaV0gPSBkZW5zZVNwZWMuYmVnaW5baV0gKyAxO1xuICAgICAgICBpZiAoeEZ3ZCA8IDAgfHwgeEZ3ZCA+PSBkaW1JKSB7XG4gICAgICAgICAgdGhyb3cgRXJyb3IoYHNsaWNlIGluZGV4ICR7ZGVuc2VTcGVjLmJlZ2luW2ldfSBvZiBkaW1lbnNpb24gJHtcbiAgICAgICAgICAgICAgaX0gb3V0IG9mIGJvdW5kcy5gKTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIHtcbiAgICAgICAgZGVuc2VTcGVjLmJlZ2luW2ldID0gY2Fub25pY2FsKFxuICAgICAgICAgICAgZGVuc2VTcGVjLmJlZ2luW2ldLCAwLCBkZW5zZVNwZWMuc3RyaWRlc1tpXSwgZGltSSwgbWFza3MsXG4gICAgICAgICAgICB2YWxpZFJhbmdlKTtcbiAgICAgICAgZGVuc2VTcGVjLmVuZFtpXSA9IGNhbm9uaWNhbChcbiAgICAgICAgICAgIGRlbnNlU3BlYy5lbmRbaV0sIDEsIGRlbnNlU3BlYy5zdHJpZGVzW2ldLCBkaW1JLCBtYXNrcywgdmFsaWRSYW5nZSk7XG4gICAgICB9XG4gICAgICAvLyBVcGRhdGUgb3B0aW1pemF0aW9uIHZhbHVlc1xuICAgICAgY29uc3QgdGFrZUFsbEluRGltZW5zaW9uID0gZGVuc2VTcGVjLnN0cmlkZXNbaV0gPT09IDEgJiZcbiAgICAgICAgICBkZW5zZVNwZWMuYmVnaW5baV0gPT09IDAgJiYgZGVuc2VTcGVjLmVuZFtpXSA9PT0gZGltSTtcbiAgICAgIGlzSWRlbnRpdHkgPSBpc0lkZW50aXR5ICYmIHRha2VBbGxJbkRpbWVuc2lvbjtcbiAgICAgIHNsaWNlRGltMCA9IHNsaWNlRGltMCAmJlxuICAgICAgICAgICgoaSA9PT0gMCAmJiBkZW5zZVNwZWMuc3RyaWRlc1tpXSA9PT0gMSkgfHwgdGFrZUFsbEluRGltZW5zaW9uKTtcbiAgICB9IGVsc2Uge1xuICAgICAgaXNJZGVudGl0eSA9XG4gICAgICAgICAgaXNJZGVudGl0eSAmJiAoKGRlbnNlU3BlYy5zdHJpZGVzW2ldID09PSAxKSAmJiBiZWdpbkFuZEVuZE1hc2tlZCk7XG4gICAgICBzbGljZURpbTAgPSBzbGljZURpbTAgJiZcbiAgICAgICAgICAoKGkgPT09IDAgJiYgZGVuc2VTcGVjLnN0cmlkZXNbaV0gPT09IDEpIHx8IGJlZ2luQW5kRW5kTWFza2VkKTtcbiAgICB9XG4gICAgLy8gQ29tcHV0ZSB0aGUgcHJvY2Vzc2luZyBzaGFwZSAodGhlIGludGVybWVkaWF0ZSBFaWdlbiB3aWxsIHByb2R1Y2UpXG4gICAgbGV0IGludGVydmFsTGVuZ3RoO1xuICAgIGxldCBrbm93bkludGVydmFsID0gZmFsc2U7XG4gICAgaWYgKGRlbnNlU3BlYy5iZWdpblZhbGlkICYmIGRlbnNlU3BlYy5lbmRWYWxpZCkge1xuICAgICAgaW50ZXJ2YWxMZW5ndGggPSBkZW5zZVNwZWMuZW5kW2ldIC0gZGVuc2VTcGVjLmJlZ2luW2ldO1xuICAgICAga25vd25JbnRlcnZhbCA9IHRydWU7XG4gICAgfSBlbHNlIGlmIChzaHJpbmtJKSB7XG4gICAgICAvLyBUaGUgZGltZW5zaW9uIGlzIHN0aWxsIGtub3duIGFzIDEgZm9yIHRoZSBwcm9jZXNzaW5nU2hhcGUsIGJ1dCB3aWxsIGJlXG4gICAgICAvLyBkaXNjYXJkZWQgZm9yIHRoZSBmaW5hbCBzaGFwZS5cbiAgICAgIGludGVydmFsTGVuZ3RoID0gMTtcbiAgICAgIGtub3duSW50ZXJ2YWwgPSB0cnVlO1xuICAgIH0gZWxzZSBpZiAoYmVnaW5BbmRFbmRNYXNrZWQpIHtcbiAgICAgIC8vIEV2ZW4gaWYgd2UgZG9uJ3QgaGF2ZSB2YWx1ZXMgZm9yIGJlZ2luIG9yIGVuZCwgd2UgZG8ga25vdyB0aGF0IHRoaXNcbiAgICAgIC8vIGRpbWVuc2lvbiBjb3ZlcnMgdGhlIHdob2xlIGludGVydmFsLiBJZiB3ZSBoYXZlIHNoYXBlIGluZm9ybWF0aW9uIGZvclxuICAgICAgLy8gdGhpcyBkaW1lbnNpb24sIHRoYXQgdGVsbHMgdXMgdGhlIGludGVydmFsIGxlbmd0aC5cbiAgICAgIGlmIChkaW1JID49IDApIHtcbiAgICAgICAgaWYgKGRlbnNlU3BlYy5zdHJpZGVzW2ldIDwgMCkge1xuICAgICAgICAgIGludGVydmFsTGVuZ3RoID0gLWRpbUk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgaW50ZXJ2YWxMZW5ndGggPSBkaW1JO1xuICAgICAgICB9XG4gICAgICAgIGtub3duSW50ZXJ2YWwgPSB0cnVlO1xuICAgICAgfVxuICAgIH1cbiAgICBpZiAoa25vd25JbnRlcnZhbCkge1xuICAgICAgbGV0IHNpemVJO1xuICAgICAgLy8gSG9sZCB6ZXJvIGlmIHRoZSBpbnRlcnZhbCBpcyBkZWdlbmVyYXRlLCBvdGhlcndpc2UgYWNjb3VudCBmb3JcbiAgICAgIC8vIHJlbWFpbmRlclxuICAgICAgaWYgKGludGVydmFsTGVuZ3RoID09PSAwIHx8XG4gICAgICAgICAgKChpbnRlcnZhbExlbmd0aCA8IDApICE9PSAoZGVuc2VTcGVjLnN0cmlkZXNbaV0gPCAwKSkpIHtcbiAgICAgICAgc2l6ZUkgPSAwO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgc2l6ZUkgPSBNYXRoLnRydW5jKGludGVydmFsTGVuZ3RoIC8gZGVuc2VTcGVjLnN0cmlkZXNbaV0pICtcbiAgICAgICAgICAgIChpbnRlcnZhbExlbmd0aCAlIGRlbnNlU3BlYy5zdHJpZGVzW2ldICE9PSAwID8gMSA6IDApO1xuICAgICAgfVxuICAgICAgcHJvY2Vzc2luZ1NoYXBlLnB1c2goc2l6ZUkpO1xuICAgIH0gZWxzZSB7XG4gICAgICBwcm9jZXNzaW5nU2hhcGUucHVzaCgtMSk7XG4gICAgfVxuICB9XG5cbiAgLy8gU3RlcCA0OiBDb21wdXRlIHRoZSBmaW5hbCBzaGFwZVxuICAvL1xuICAvLyBuZXdBeGlzIHdpbGwgaW5jcmVhc2UgZGltZW5zaW9uIGJ5IDEgKHdpdGggYSBvbmUtc2l6ZSBkaW1lbnNpb24pXG4gIC8vIHNsaWNlcyBsaWtlIGZvb1szLCAuLi5dIHdpbGwgcmVkdWNlIGRpbWVuc2lvbiBieSAxLlxuICAvLyBUaGlzIGNhbm5vdCBiZSBkb25lIGVhcmxpZXIsIGJlY2F1c2UgaXQgZGVwZW5kcyBvbiBTdGVwIDMuXG4gIGZvciAobGV0IGRlbnNlRGltID0gMDsgZGVuc2VEaW0gPCBkZW5zZVNwZWMuZmluYWxTaGFwZUdhdGhlckluZGljZXMubGVuZ3RoO1xuICAgICAgICsrZGVuc2VEaW0pIHtcbiAgICBjb25zdCBnYXRoZXJJbmRleCA9IGRlbnNlU3BlYy5maW5hbFNoYXBlR2F0aGVySW5kaWNlc1tkZW5zZURpbV07XG4gICAgaWYgKGdhdGhlckluZGV4ID49IDApIHtcbiAgICAgIGZpbmFsU2hhcGUucHVzaChwcm9jZXNzaW5nU2hhcGVbZ2F0aGVySW5kZXhdKTtcbiAgICB9IGVsc2UgaWYgKGdhdGhlckluZGV4ID09PSBORVdfQVhJUykge1xuICAgICAgZmluYWxTaGFwZS5wdXNoKDEpO1xuICAgIH1cbiAgfVxuXG4gIGNvbnN0IGZpbmFsU2hhcGVTcGFyc2UgPSBmaW5hbFNoYXBlLmZpbHRlcihcbiAgICAgIChkaW0sIGkpID0+IGRlbnNlU3BlYy5maW5hbFNoYXBlR2F0aGVySW5kaWNlc1tpXSAhPT0gTkVXX0FYSVMpO1xuXG4gIHJldHVybiB7XG4gICAgZmluYWxTaGFwZVNwYXJzZSxcbiAgICBmaW5hbFNoYXBlLFxuICAgIGlzSWRlbnRpdHksXG4gICAgc2xpY2VEaW0wLFxuICAgIGlzU2ltcGxlU2xpY2UsXG4gICAgYmVnaW46IGRlbnNlU3BlYy5iZWdpbixcbiAgICBlbmQ6IGRlbnNlU3BlYy5lbmQsXG4gICAgc3RyaWRlczogZGVuc2VTcGVjLnN0cmlkZXNcbiAgfTtcbn1cblxuZnVuY3Rpb24gYnVpbGREZW5zZVNwZWMoXG4gICAgc3BhcnNlOiBTdHJpZGVkU2xpY2VTcGFyc2VTcGVjLCBkZW5zZTogU3RyaWRlZFNsaWNlRGVuc2VTcGVjKSB7XG4gIGRlbnNlLmJlZ2luTWFzayA9IDA7XG4gIGRlbnNlLmVuZE1hc2sgPSAwO1xuICBkZW5zZS5zaHJpbmtBeGlzTWFzayA9IDA7XG5cbiAgbGV0IGZ1bGxJbmRleCA9IDA7XG4gIGRlbnNlLmJlZ2luVmFsaWQgPSBzcGFyc2UuYmVnaW4gIT0gbnVsbDtcbiAgZGVuc2UuZW5kVmFsaWQgPSBzcGFyc2UuZW5kICE9IG51bGw7XG5cbiAgZGVuc2UuYmVnaW4gPSBuZXcgQXJyYXkoZGVuc2UuZGltcyk7XG4gIGRlbnNlLmVuZCA9IG5ldyBBcnJheShkZW5zZS5kaW1zKTtcbiAgZGVuc2Uuc3RyaWRlcyA9IG5ldyBBcnJheShkZW5zZS5kaW1zKTtcbiAgZGVuc2UuZmluYWxTaGFwZUdhdGhlckluZGljZXMgPSBbXTtcbiAgZGVuc2UuZmluYWxTaGFwZUdhdGhlckluZGljZXNTcGFyc2UgPSBbXTtcbiAgZGVuc2UuaW5wdXRTaGFwZUdhdGhlckluZGljZXNTcGFyc2UgPSBuZXcgQXJyYXkoZGVuc2UuZGltcyk7XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBzcGFyc2UuZGltczsgaSsrKSB7XG4gICAgaWYgKCgxIDw8IGkpICYgc3BhcnNlLmVsbGlwc2lzTWFzaykge1xuICAgICAgLy8gT25seSB0aGUgYml0IHRoYXQgaGFzIGVsbGlwc2lzIHdpbGwgZmFsbCBpbiB0aGlzIGNvbmRpdGlvbi5cbiAgICAgIC8vIEV4cGFuZCB0aGUgZWxsaXBzaXMgaW50byB0aGUgYXBwcm9wcmlhdGUgaW5kaWNlc1xuICAgICAgLy8gTm90ZTogdGhpcyBvbmx5IHdvcmtzIGJlY2F1c2Ugd2UgZ3VhcmFudGVlZCBvbmUgZWxsaXBzaXMuXG4gICAgICBjb25zdCBuZXh0SW5kZXggPSBNYXRoLm1pbihcbiAgICAgICAgICBkZW5zZS5kaW1zIC0gKHNwYXJzZS5kaW1zIC0gaSkgKyAxICsgc3BhcnNlLm51bUFkZEF4aXNBZnRlckVsbGlwc2lzLFxuICAgICAgICAgIGRlbnNlLmRpbXMpO1xuICAgICAgZm9yICg7IGZ1bGxJbmRleCA8IG5leHRJbmRleDsgZnVsbEluZGV4KyspIHtcbiAgICAgICAgLy8gbmV3QXhpcyBhcmVuJ3QgcmVhbCBheGlzIHNvIHlvdSBoYXZlIHRvIHNraXAuXG4gICAgICAgIGRlbnNlLmJlZ2luW2Z1bGxJbmRleF0gPSAwO1xuICAgICAgICBkZW5zZS5lbmRbZnVsbEluZGV4XSA9IDA7XG4gICAgICAgIGRlbnNlLnN0cmlkZXNbZnVsbEluZGV4XSA9IDE7XG4gICAgICAgIGRlbnNlLmJlZ2luTWFzayB8PSAoMSA8PCBmdWxsSW5kZXgpO1xuICAgICAgICBkZW5zZS5lbmRNYXNrIHw9ICgxIDw8IGZ1bGxJbmRleCk7XG4gICAgICAgIGRlbnNlLmZpbmFsU2hhcGVHYXRoZXJJbmRpY2VzLnB1c2goZnVsbEluZGV4KTtcbiAgICAgICAgZGVuc2UuZmluYWxTaGFwZUdhdGhlckluZGljZXNTcGFyc2UucHVzaCgtMSk7XG4gICAgICAgIGRlbnNlLmlucHV0U2hhcGVHYXRoZXJJbmRpY2VzU3BhcnNlW2Z1bGxJbmRleF0gPSBpO1xuICAgICAgfVxuICAgIH0gZWxzZSBpZiAoKDEgPDwgaSkgJiBzcGFyc2UubmV3QXhpc01hc2spIHtcbiAgICAgIC8vIE9ubHkgdGhlIGJpdCB0aGF0IGhhcyBuZXdBeGlzIHdpbGwgZmFsbCBpbiB0aGlzIGNvbmRpdGlvbi5cbiAgICAgIGRlbnNlLmZpbmFsU2hhcGVHYXRoZXJJbmRpY2VzLnB1c2goTkVXX0FYSVMpO1xuICAgICAgZGVuc2UuZmluYWxTaGFwZUdhdGhlckluZGljZXNTcGFyc2UucHVzaCgtMSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGlmIChmdWxsSW5kZXggPT09IGRlbnNlLmJlZ2luLmxlbmd0aCkge1xuICAgICAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgICAgIGBJbmRleCBvdXQgb2YgcmFuZ2UgdXNpbmcgaW5wdXQgZGltICR7ZnVsbEluZGV4fTsgaW5wdXQgYCArXG4gICAgICAgICAgICBgaGFzIG9ubHkgJHtkZW5zZS5kaW1zfSBkaW1zLCAke2RlbnNlLmJlZ2luLmxlbmd0aH0uYCk7XG4gICAgICB9XG5cbiAgICAgIC8vIEdhdGhlciBzbGljaW5nIHNwZWMgaW50byBhcHByb3ByaWF0ZSBpbmRleC5cbiAgICAgIGlmIChzcGFyc2UuYmVnaW4gIT0gbnVsbCkge1xuICAgICAgICBkZW5zZS5iZWdpbltmdWxsSW5kZXhdID0gc3BhcnNlLmJlZ2luW2ldO1xuICAgICAgfVxuICAgICAgaWYgKHNwYXJzZS5lbmQgIT0gbnVsbCkge1xuICAgICAgICBkZW5zZS5lbmRbZnVsbEluZGV4XSA9IHNwYXJzZS5lbmRbaV07XG4gICAgICB9XG4gICAgICBkZW5zZS5zdHJpZGVzW2Z1bGxJbmRleF0gPSBzcGFyc2Uuc3RyaWRlc1tpXTtcbiAgICAgIGlmIChzcGFyc2UuYmVnaW5NYXNrICYgKDEgPDwgaSkpIHtcbiAgICAgICAgZGVuc2UuYmVnaW5NYXNrIHw9ICgxIDw8IGZ1bGxJbmRleCk7XG4gICAgICB9XG4gICAgICBpZiAoc3BhcnNlLmVuZE1hc2sgJiAoMSA8PCBpKSkge1xuICAgICAgICBkZW5zZS5lbmRNYXNrIHw9ICgxIDw8IGZ1bGxJbmRleCk7XG4gICAgICB9XG4gICAgICAvLyBJZiBzaHJpbmssIHJlY29yZCB3aGVyZSB0byBnZXQgdGhlIGRpbWVuc2lvbmFsaXR5IGZyb20gKGkuZS4gbmV3QXhpcylcbiAgICAgIC8vIGNyZWF0ZXMgYSBmYWtlIDEgc2l6ZSBkaW1lbnNpb24uIEFsc28gcmVtZW1iZXIgc2hyaW5rIGF4aXMgKG5vdyBpblxuICAgICAgLy8gZGVuc2UgZm9ybSkgc28gd2UgY2FuIGlnbm9yZSBkZW5zZS5lbmQgYmVsb3cuXG4gICAgICBpZiAoc3BhcnNlLnNocmlua0F4aXNNYXNrICYgKDEgPDwgaSkpIHtcbiAgICAgICAgZGVuc2UuZmluYWxTaGFwZUdhdGhlckluZGljZXMucHVzaChTSFJJTktfQVhJUyk7XG4gICAgICAgIGRlbnNlLmZpbmFsU2hhcGVHYXRoZXJJbmRpY2VzU3BhcnNlLnB1c2goLTEpO1xuICAgICAgICBkZW5zZS5zaHJpbmtBeGlzTWFzayB8PSAoMSA8PCBmdWxsSW5kZXgpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgZGVuc2UuZmluYWxTaGFwZUdhdGhlckluZGljZXMucHVzaChmdWxsSW5kZXgpO1xuICAgICAgICAvLyBSZW1lbWJlciB0aGF0IHdoZXJlIGluIHRoZSBzcGFyc2Ugc2hhcGUgdGhlIGRlbnNlIGRpbSBjb21lcyBmcm9tLlxuICAgICAgICBkZW5zZS5maW5hbFNoYXBlR2F0aGVySW5kaWNlc1NwYXJzZS5wdXNoKGkpO1xuICAgICAgfVxuICAgICAgZGVuc2UuaW5wdXRTaGFwZUdhdGhlckluZGljZXNTcGFyc2VbZnVsbEluZGV4XSA9IGk7XG4gICAgICBmdWxsSW5kZXgrKztcbiAgICB9XG4gIH1cbn1cblxuZnVuY3Rpb24gY2Fub25pY2FsKFxuICAgIHg6IG51bWJlciwgYzogbnVtYmVyLCBzdHJpZGVJOiBudW1iZXIsIGRpbUk6IG51bWJlciwgbWFza3M6IG51bWJlcltdLFxuICAgIHZhbGlkUmFuZ2U6IG51bWJlcltdKSB7XG4gIGlmIChtYXNrc1tjXSkge1xuICAgIHJldHVybiBzdHJpZGVJID4gMCA/IHZhbGlkUmFuZ2VbY10gOiB2YWxpZFJhbmdlWyhjICsgMSkgJiAxXTtcbiAgfSBlbHNlIHtcbiAgICBjb25zdCB4RndkID0geCA8IDAgPyBkaW1JICsgeCA6IHg7ICAvLyBtYWtlIG5lZ2F0aXZlIGluZGljZXMgcG9zaXRpdmVcbiAgICByZXR1cm4geEZ3ZCA8IHZhbGlkUmFuZ2VbMF0gPyB2YWxpZFJhbmdlWzBdIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB4RndkID4gdmFsaWRSYW5nZVsxXSA/IHZhbGlkUmFuZ2VbMV0gOiB4RndkO1xuICB9XG59XG4iXX0=