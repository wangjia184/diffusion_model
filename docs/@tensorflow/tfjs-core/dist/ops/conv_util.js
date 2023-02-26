/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
/**
 *
 * @param inputShape Input tensor shape is of the following dimensions:
 *     `[batch, height, width, inChannels]`.
 * @param filterShape The filter shape is of the following dimensions:
 *     `[filterHeight, filterWidth, depth]`.
 * @param strides The strides of the sliding window for each dimension of the
 *     input tensor: `[strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat The data format of the input and output data.
 *     Defaults to 'NHWC'.
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`.
 *     Defaults to `[1, 1]`. If `dilations` is a single number, then
 *     `dilationHeight == dilationWidth`.
 */
export function computeDilation2DInfo(inputShape, filterShape, strides, pad, dataFormat = 'NHWC', dilations) {
    // `computerConv2DInfo` require filterShape to be in the dimension of:
    // `[filterHeight, filterWidth, depth, outDepth]`, dilation2d doesn't have
    // outDepth, it should have the same depth as the input.
    // Input shape: [batch, height, width, inChannels]
    const inputChannels = inputShape[3];
    const $filterShape = [...filterShape, inputChannels];
    const $dataFormat = convertConv2DDataFormat(dataFormat);
    return computeConv2DInfo(inputShape, $filterShape, strides, dilations, pad, null /* roundingMode */, null /* depthWise */, $dataFormat);
}
export function computePool2DInfo(inShape, filterSize, strides, dilations, pad, roundingMode, dataFormat = 'channelsLast') {
    const [filterHeight, filterWidth] = parseTupleParam(filterSize);
    let filterShape;
    if (dataFormat === 'channelsLast') {
        filterShape = [filterHeight, filterWidth, inShape[3], inShape[3]];
    }
    else if (dataFormat === 'channelsFirst') {
        filterShape = [filterHeight, filterWidth, inShape[1], inShape[1]];
    }
    else {
        throw new Error(`Unknown dataFormat ${dataFormat}`);
    }
    return computeConv2DInfo(inShape, filterShape, strides, dilations, pad, roundingMode, false, dataFormat);
}
/**
 * Computes the information for a forward pass of a pooling3D operation.
 */
export function computePool3DInfo(inShape, filterSize, strides, dilations, pad, roundingMode, dataFormat = 'NDHWC') {
    const [filterDepth, filterHeight, filterWidth] = parse3TupleParam(filterSize);
    let filterShape;
    let $dataFormat;
    if (dataFormat === 'NDHWC') {
        $dataFormat = 'channelsLast';
        filterShape =
            [filterDepth, filterHeight, filterWidth, inShape[4], inShape[4]];
    }
    else if (dataFormat === 'NCDHW') {
        $dataFormat = 'channelsFirst';
        filterShape =
            [filterDepth, filterHeight, filterWidth, inShape[1], inShape[1]];
    }
    else {
        throw new Error(`Unknown dataFormat ${dataFormat}`);
    }
    return computeConv3DInfo(inShape, filterShape, strides, dilations, pad, false, $dataFormat, roundingMode);
}
/**
 * Computes the information for a forward pass of a convolution/pooling
 * operation.
 */
export function computeConv2DInfo(inShape, filterShape, strides, dilations, pad, roundingMode, depthwise = false, dataFormat = 'channelsLast') {
    let [batchSize, inHeight, inWidth, inChannels] = [-1, -1, -1, -1];
    if (dataFormat === 'channelsLast') {
        [batchSize, inHeight, inWidth, inChannels] = inShape;
    }
    else if (dataFormat === 'channelsFirst') {
        [batchSize, inChannels, inHeight, inWidth] = inShape;
    }
    else {
        throw new Error(`Unknown dataFormat ${dataFormat}`);
    }
    const [filterHeight, filterWidth, , filterChannels] = filterShape;
    const [strideHeight, strideWidth] = parseTupleParam(strides);
    const [dilationHeight, dilationWidth] = parseTupleParam(dilations);
    const effectiveFilterHeight = getEffectiveFilterSize(filterHeight, dilationHeight);
    const effectiveFilterWidth = getEffectiveFilterSize(filterWidth, dilationWidth);
    const { padInfo, outHeight, outWidth } = getPadAndOutInfo(pad, inHeight, inWidth, strideHeight, strideWidth, effectiveFilterHeight, effectiveFilterWidth, roundingMode, dataFormat);
    const outChannels = depthwise ? filterChannels * inChannels : filterChannels;
    let outShape;
    if (dataFormat === 'channelsFirst') {
        outShape = [batchSize, outChannels, outHeight, outWidth];
    }
    else if (dataFormat === 'channelsLast') {
        outShape = [batchSize, outHeight, outWidth, outChannels];
    }
    return {
        batchSize,
        dataFormat,
        inHeight,
        inWidth,
        inChannels,
        outHeight,
        outWidth,
        outChannels,
        padInfo,
        strideHeight,
        strideWidth,
        filterHeight,
        filterWidth,
        effectiveFilterHeight,
        effectiveFilterWidth,
        dilationHeight,
        dilationWidth,
        inShape,
        outShape,
        filterShape
    };
}
/**
 * Computes the information for a forward pass of a 3D convolution/pooling
 * operation.
 */
export function computeConv3DInfo(inShape, filterShape, strides, dilations, pad, depthwise = false, dataFormat = 'channelsLast', roundingMode) {
    let [batchSize, inDepth, inHeight, inWidth, inChannels] = [-1, -1, -1, -1, -1];
    if (dataFormat === 'channelsLast') {
        [batchSize, inDepth, inHeight, inWidth, inChannels] = inShape;
    }
    else if (dataFormat === 'channelsFirst') {
        [batchSize, inChannels, inDepth, inHeight, inWidth] = inShape;
    }
    else {
        throw new Error(`Unknown dataFormat ${dataFormat}`);
    }
    const [filterDepth, filterHeight, filterWidth, , filterChannels] = filterShape;
    const [strideDepth, strideHeight, strideWidth] = parse3TupleParam(strides);
    const [dilationDepth, dilationHeight, dilationWidth] = parse3TupleParam(dilations);
    const effectiveFilterDepth = getEffectiveFilterSize(filterDepth, dilationDepth);
    const effectiveFilterHeight = getEffectiveFilterSize(filterHeight, dilationHeight);
    const effectiveFilterWidth = getEffectiveFilterSize(filterWidth, dilationWidth);
    const { padInfo, outDepth, outHeight, outWidth } = get3DPadAndOutInfo(pad, inDepth, inHeight, inWidth, strideDepth, strideHeight, strideWidth, effectiveFilterDepth, effectiveFilterHeight, effectiveFilterWidth, roundingMode);
    const outChannels = depthwise ? filterChannels * inChannels : filterChannels;
    let outShape;
    if (dataFormat === 'channelsFirst') {
        outShape = [batchSize, outChannels, outDepth, outHeight, outWidth];
    }
    else if (dataFormat === 'channelsLast') {
        outShape = [batchSize, outDepth, outHeight, outWidth, outChannels];
    }
    return {
        batchSize,
        dataFormat,
        inDepth,
        inHeight,
        inWidth,
        inChannels,
        outDepth,
        outHeight,
        outWidth,
        outChannels,
        padInfo,
        strideDepth,
        strideHeight,
        strideWidth,
        filterDepth,
        filterHeight,
        filterWidth,
        effectiveFilterDepth,
        effectiveFilterHeight,
        effectiveFilterWidth,
        dilationDepth,
        dilationHeight,
        dilationWidth,
        inShape,
        outShape,
        filterShape
    };
}
function computeOutputShape2D(inShape, fieldSize, stride, zeroPad, roundingMode) {
    if (zeroPad == null) {
        zeroPad = computeDefaultPad(inShape, fieldSize, stride);
    }
    const inputRows = inShape[0];
    const inputCols = inShape[1];
    const outputRows = round((inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
    const outputCols = round((inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
    return [outputRows, outputCols];
}
function computeOutputShape4D(inShape, filterShape, outChannels, strides, zeroPad, roundingMode) {
    if (zeroPad == null) {
        zeroPad = computeDefaultPad(inShape, filterShape[0], strides[0]);
    }
    const outShape = [0, 0, 0, outChannels];
    for (let index = 0; index < 3; index++) {
        if (inShape[index] + 2 * zeroPad >= filterShape[index]) {
            outShape[index] = round((inShape[index] - filterShape[index] + 2 * zeroPad) / strides[index] +
                1, roundingMode);
        }
    }
    return outShape;
}
export function computeDefaultPad(inputShape, fieldSize, stride, dilation = 1) {
    const effectiveFieldSize = getEffectiveFilterSize(fieldSize, dilation);
    return Math.floor((inputShape[0] * (stride - 1) - stride + effectiveFieldSize) / 2);
}
function parseTupleParam(param) {
    if (typeof param === 'number') {
        return [param, param, param];
    }
    if (param.length === 2) {
        return [param[0], param[1], 1];
    }
    return param;
}
function parse3TupleParam(param) {
    return typeof param === 'number' ? [param, param, param] : param;
}
/* See https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
 * Atrous convolution is equivalent to standard convolution with upsampled
 * filters with effective_filter_height =
 * filter_height + (filter_height - 1) * (dilation - 1)
 * and effective_filter_width =
 * filter_width + (filter_width - 1) * (dilation - 1),
 * produced by inserting dilation - 1 zeros along consecutive elements across
 * the filters' spatial dimensions.
 * When there is a dilation, this converts a filter dimension to the
 * effective filter dimension, so it can be used in a standard convolution.
 */
function getEffectiveFilterSize(filterSize, dilation) {
    if (dilation <= 1) {
        return filterSize;
    }
    return filterSize + (filterSize - 1) * (dilation - 1);
}
function getPadAndOutInfo(pad, inHeight, inWidth, strideHeight, strideWidth, filterHeight, filterWidth, roundingMode, dataFormat) {
    let padInfo;
    let outHeight;
    let outWidth;
    if (typeof pad === 'number') {
        const padType = (pad === 0) ? 'VALID' : 'NUMBER';
        padInfo = { top: pad, bottom: pad, left: pad, right: pad, type: padType };
        const outShape = computeOutputShape2D([inHeight, inWidth], filterHeight, strideHeight, pad, roundingMode);
        outHeight = outShape[0];
        outWidth = outShape[1];
    }
    else if (pad === 'same') {
        outHeight = Math.ceil(inHeight / strideHeight);
        outWidth = Math.ceil(inWidth / strideWidth);
        const padAlongHeight = Math.max(0, (outHeight - 1) * strideHeight + filterHeight - inHeight);
        const padAlongWidth = Math.max(0, (outWidth - 1) * strideWidth + filterWidth - inWidth);
        const top = Math.floor(padAlongHeight / 2);
        const bottom = padAlongHeight - top;
        const left = Math.floor(padAlongWidth / 2);
        const right = padAlongWidth - left;
        padInfo = { top, bottom, left, right, type: 'SAME' };
    }
    else if (pad === 'valid') {
        padInfo = { top: 0, bottom: 0, left: 0, right: 0, type: 'VALID' };
        outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
        outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
    }
    else if (typeof pad === 'object') {
        const top = dataFormat === 'channelsLast' ? pad[1][0] : pad[2][0];
        const bottom = dataFormat === 'channelsLast' ? pad[1][1] : pad[2][1];
        const left = dataFormat === 'channelsLast' ? pad[2][0] : pad[3][0];
        const right = dataFormat === 'channelsLast' ? pad[2][1] : pad[3][1];
        const padType = (top === 0 && bottom === 0 && left === 0 && right === 0) ?
            'VALID' :
            'EXPLICIT';
        padInfo = { top, bottom, left, right, type: padType };
        outHeight = round((inHeight - filterHeight + top + bottom) / strideHeight + 1, roundingMode);
        outWidth = round((inWidth - filterWidth + left + right) / strideWidth + 1, roundingMode);
    }
    else {
        throw Error(`Unknown padding parameter: ${pad}`);
    }
    return { padInfo, outHeight, outWidth };
}
function get3DPadAndOutInfo(pad, inDepth, inHeight, inWidth, strideDepth, strideHeight, strideWidth, filterDepth, filterHeight, filterWidth, roundingMode) {
    let padInfo;
    let outDepth;
    let outHeight;
    let outWidth;
    if (pad === 'valid') {
        pad = 0;
    }
    if (typeof pad === 'number') {
        const padType = (pad === 0) ? 'VALID' : 'NUMBER';
        padInfo = {
            top: pad,
            bottom: pad,
            left: pad,
            right: pad,
            front: pad,
            back: pad,
            type: padType
        };
        const outShape = computeOutputShape4D([inDepth, inHeight, inWidth, 1], [filterDepth, filterHeight, filterWidth], 1, [strideDepth, strideHeight, strideWidth], pad, roundingMode);
        outDepth = outShape[0];
        outHeight = outShape[1];
        outWidth = outShape[2];
    }
    else if (pad === 'same') {
        outDepth = Math.ceil(inDepth / strideDepth);
        outHeight = Math.ceil(inHeight / strideHeight);
        outWidth = Math.ceil(inWidth / strideWidth);
        const padAlongDepth = (outDepth - 1) * strideDepth + filterDepth - inDepth;
        const padAlongHeight = (outHeight - 1) * strideHeight + filterHeight - inHeight;
        const padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
        const front = Math.floor(padAlongDepth / 2);
        const back = padAlongDepth - front;
        const top = Math.floor(padAlongHeight / 2);
        const bottom = padAlongHeight - top;
        const left = Math.floor(padAlongWidth / 2);
        const right = padAlongWidth - left;
        padInfo = { top, bottom, left, right, front, back, type: 'SAME' };
    }
    else {
        throw Error(`Unknown padding parameter: ${pad}`);
    }
    return { padInfo, outDepth, outHeight, outWidth };
}
/**
 * Rounds a value depending on the rounding mode
 * @param value
 * @param roundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function round(value, roundingMode) {
    if (!roundingMode) {
        return Math.trunc(value);
    }
    switch (roundingMode) {
        case 'round':
            // used for Caffe Conv
            return Math.round(value);
        case 'ceil':
            // used for Caffe Pool
            return Math.ceil(value);
        case 'floor':
            return Math.floor(value);
        default:
            throw new Error(`Unknown roundingMode ${roundingMode}`);
    }
}
export function tupleValuesAreOne(param) {
    const [dimA, dimB, dimC] = parseTupleParam(param);
    return dimA === 1 && dimB === 1 && dimC === 1;
}
export function eitherStridesOrDilationsAreOne(strides, dilations) {
    return tupleValuesAreOne(strides) || tupleValuesAreOne(dilations);
}
export function stridesOrDilationsArePositive(values) {
    return parseTupleParam(values).every(value => value > 0);
}
/**
 * Convert Conv2D dataFormat from 'NHWC'|'NCHW' to
 *    'channelsLast'|'channelsFirst'
 * @param dataFormat in 'NHWC'|'NCHW' mode
 * @return dataFormat in 'channelsLast'|'channelsFirst' mode
 * @throws unknown dataFormat
 */
export function convertConv2DDataFormat(dataFormat) {
    if (dataFormat === 'NHWC') {
        return 'channelsLast';
    }
    else if (dataFormat === 'NCHW') {
        return 'channelsFirst';
    }
    else {
        throw new Error(`Unknown dataFormat ${dataFormat}`);
    }
}
/**
 * Check validity of pad when using dimRoundingMode.
 * @param opDesc A string of op description
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid` output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @throws unknown padding parameter
 */
export function checkPadOnDimRoundingMode(opDesc, pad, dimRoundingMode) {
    if (dimRoundingMode != null) {
        if (typeof pad === 'string') {
            throw Error(`Error in ${opDesc}: pad must be an integer when using ` +
                `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
        }
        else if (typeof pad === 'number') {
            util.assert(util.isInt(pad), () => `Error in ${opDesc}: pad must be an integer when using ` +
                `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
        }
        else if (typeof pad === 'object') {
            pad.forEach(p => {
                p.forEach(v => {
                    util.assert(util.isInt(v), () => `Error in ${opDesc}: pad must be an integer when using ` +
                        `dimRoundingMode ${dimRoundingMode} but got pad ${v}.`);
                });
            });
        }
        else {
            throw Error(`Error in ${opDesc}: Unknown padding parameter: ${pad}`);
        }
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udl91dGlsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvY29udl91dGlsLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBMERoQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0F1Qkc7QUFDSCxNQUFNLFVBQVUscUJBQXFCLENBQ2pDLFVBQTRDLEVBQzVDLFdBQXFDLEVBQUUsT0FBZ0MsRUFDdkUsR0FBMEIsRUFBRSxhQUFxQixNQUFNLEVBQ3ZELFNBQWtDO0lBQ3BDLHNFQUFzRTtJQUN0RSwwRUFBMEU7SUFDMUUsd0RBQXdEO0lBQ3hELGtEQUFrRDtJQUNsRCxNQUFNLGFBQWEsR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDcEMsTUFBTSxZQUFZLEdBQ2QsQ0FBQyxHQUFHLFdBQVcsRUFBRSxhQUFhLENBQXFDLENBQUM7SUFDeEUsTUFBTSxXQUFXLEdBQUcsdUJBQXVCLENBQUMsVUFBVSxDQUFDLENBQUM7SUFFeEQsT0FBTyxpQkFBaUIsQ0FDcEIsVUFBVSxFQUFFLFlBQVksRUFBRSxPQUFPLEVBQUUsU0FBUyxFQUFFLEdBQUcsRUFDakQsSUFBSSxDQUFDLGtCQUFrQixFQUFFLElBQUksQ0FBQyxlQUFlLEVBQUUsV0FBVyxDQUFDLENBQUM7QUFDbEUsQ0FBQztBQUVELE1BQU0sVUFBVSxpQkFBaUIsQ0FDN0IsT0FBeUMsRUFDekMsVUFBbUMsRUFBRSxPQUFnQyxFQUNyRSxTQUFrQyxFQUNsQyxHQUEwQyxFQUMxQyxZQUFxQyxFQUNyQyxhQUE2QyxjQUFjO0lBQzdELE1BQU0sQ0FBQyxZQUFZLEVBQUUsV0FBVyxDQUFDLEdBQUcsZUFBZSxDQUFDLFVBQVUsQ0FBQyxDQUFDO0lBRWhFLElBQUksV0FBNkMsQ0FBQztJQUNsRCxJQUFJLFVBQVUsS0FBSyxjQUFjLEVB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LE1BQU0sRUFBRSxDQUFDLEVBQUUsSUFBSSxFQUFFLENBQUMsRUFBRSxLQUFLLEVBQUUsQ0FBQyxFQUFFLElBQUksRUFBRSxPQUFPLEVBQUMsQ0FBQztRQUNoRSxTQUFTLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLFFBQVEsR0FBRyxZQUFZLEdBQUcsQ0FBQyxDQUFDLEdBQUcsWUFBWSxDQUFDLENBQUM7UUFDcEUsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxPQUFPLEdBQUcsV0FBVyxHQUFHLENBQUMsQ0FBQyxHQUFHLFdBQVcsQ0FBQyxDQUFDO0tBQ2pFO1NBQU0sSUFBSSxPQUFPLEdBQUcsS0FBSyxRQUFRLEVBQUU7UUFDbEMsTUFBTSxHQUFHLEdBQUcsVUFBVSxLQUFLLGNBQWMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEUsTUFBTSxNQUFNLEdBQUcsVUFBVSxLQUFLLGNBQWMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDckUsTUFBTSxJQUFJLEdBQUcsVUFBVSxLQUFLLGNBQWMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbkUsTUFBTSxLQUFLLEdBQUcsVUFBVSxLQUFLLGNBQWMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDcEUsTUFBTSxPQUFPLEdBQUcsQ0FBQyxH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O0FBRUQ7Ozs7Ozs7Ozs7Ozs7O0dBY0c7QUFDSCxNQUFNLFVBQVUseUJBQXlCLENBQ3JDLE1BQWMsRUFBRSxHQUEwQyxFQUMxRCxlQUF3QztJQUMxQyxJQUFJLGVBQWUsSUFBSSxJQUFJLEVBQUU7UUFDM0IsSUFBSSxPQUFPLEdBQUcsS0FBSyxRQUFRLEVBQUU7WUFDM0IsTUFBTSxLQUFLLENBQ1AsWUFBWSxNQUFNLHNDQUFzQztnQkFDeEQsbUJBQW1CLGVBQWUsZ0JBQWdCLEdBQUcsR0FBRyxDQUFDLENBQUM7U0FDL0Q7YUFBTSxJQUFJLE9BQU8sR0FBRyxLQUFLLFFBQVEsRUFBRTtZQUNsQyxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksQ0FBQyxLQUFLLENBQUMsR0FBRyxDQUFDLEVBQ2YsR0FBRyxFQUFFLENBQUMsWUFBWSxNQUFNLHNDQUFzQztnQkFDMUQsbUJBQW1CLGVBQWUsZ0JBQWdCLEdBQUcsR0FBRyxDQUFDLENBQUM7U0FDbkU7YUFBTSxJQUFJLE9BQU8sR0FBRyxLQUFLLFFBQVEsRUFBRTtZQUNqQyxHQUF1QixDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRTtnQkFDbkMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRTtvQkFDWixJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQ2IsR0FBRyxFQUFFLENBQUMsWUFBWSxNQUFNLHNDQUFzQzt3QkFDMUQsbUJBQW1CLGVBQWUsZ0JBQWdCLENBQUMsR0FBRyxDQUFDLENBQUM7Z0JBQ2xFLENBQUMsQ0FBQyxDQUFDO1lBQ0wsQ0FBQyxDQUFDLENBQUM7U0FDSjthQUFNO1lBQ0wsTUFBTSxLQUFLLENBQUMsWUFBWSxNQUFNLGdDQUFnQyxHQUFHLEVBQUUsQ0FBQyxDQUFDO1NBQ3RFO0tBQ0Y7QUFDSCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG50eXBlIFBhZFR5cGUgPSAnU0FNRSd8J1ZBTElEJ3wnTlVNQkVSJ3wnRVhQTElDSVQnO1xuXG4vLyBGb3IgTkhXQyBzaG91bGQgYmUgaW4gdGhlIGZvbGxvd2luZyBmb3JtOlxuLy8gIFtbMCwgMF0sIFtwYWRfdG9wLHBhZF9ib3R0b21dLCBbcGFkX2xlZnQsIHBhZF9yaWdodF0sIFswLCAwXV1cbi8vIEZvciBOQ0hXIHNob3VsZCBiZSBpbiB0aGUgZm9sbG93aW5nIGZvcm06XG4vLyAgW1swLCAwXSwgWzAsIDBdLCBbcGFkX3RvcCxwYWRfYm90dG9tXSwgW3BhZF9sZWZ0LCBwYWRfcmlnaHRdXVxuLy8gUmVmZXJlbmNlOiBodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udjJkXG5leHBvcnQgdHlwZSBFeHBsaWNpdFBhZGRpbmcgPVxuICAgIFtbbnVtYmVyLCBudW1iZXJdLCBbbnVtYmVyLCBudW1iZXJdLCBbbnVtYmVyLCBudW1iZXJdLCBbbnVtYmVyLCBudW1iZXJdXTtcblxuZXhwb3J0IHR5cGUgUGFkSW5mbyA9IHtcbiAgdG9wOiBudW1iZXIsXG4gIGxlZnQ6IG51bWJlcixcbiAgcmlnaHQ6IG51bWJlcixcbiAgYm90dG9tOiBudW1iZXIsXG4gIHR5cGU6IFBhZFR5cGVcbn07XG5cbmV4cG9ydCB0eXBlIFBhZEluZm8zRCA9IHtcbiAgdG9wOiBudW1iZXIsXG4gIGxlZnQ6IG51bWJlcixcbiAgcmlnaHQ6IG51bWJlcixcbiAgYm90dG9tOiBudW1iZXIsXG4gIGZyb250OiBudW1iZXIsXG4gIGJhY2s6IG51bWJlcixcbiAgdHlwZTogUGFkVHlwZVxufTtcblxuLyoqXG4gKiBJbmZvcm1hdGlvbiBhYm91dCB0aGUgZm9yd2FyZCBwYXNzIG9mIGEgY29udm9sdXRpb24vcG9vbGluZyBvcGVyYXRpb24uXG4gKiBJdCBpbmNsdWRlcyBpbnB1dCBhbmQgb3V0cHV0IHNoYXBlLCBzdHJpZGVzLCBmaWx0ZXIgc2l6ZSBhbmQgcGFkZGluZ1xuICogaW5mb3JtYXRpb24uXG4gKi9cbmV4cG9ydCB0eXBlIENvbnYyREluZm8gPSB7XG4gIGJhdGNoU2l6ZTogbnVtYmVyLFxuICBpbkhlaWdodDogbnVtYmVyLFxuICBpbldpZHRoOiBudW1iZXIsXG4gIGluQ2hhbm5lbHM6IG51bWJlcixcbiAgb3V0SGVpZ2h0OiBudW1iZXIsXG4gIG91dFdpZHRoOiBudW1iZXIsXG4gIG91dENoYW5uZWxzOiBudW1iZXIsXG4gIGRhdGFGb3JtYXQ6ICdjaGFubmVsc0ZpcnN0J3wnY2hhbm5lbHNMYXN0JyxcbiAgc3RyaWRlSGVpZ2h0OiBudW1iZXIsXG4gIHN0cmlkZVdpZHRoOiBudW1iZXIsXG4gIGRpbGF0aW9uSGVpZ2h0OiBudW1iZXIsXG4gIGRpbGF0aW9uV2lkdGg6IG51bWJlcixcbiAgZmlsdGVySGVpZ2h0OiBudW1iZXIsXG4gIGZpbHRlcldpZHRoOiBudW1iZXIsXG4gIGVmZmVjdGl2ZUZpbHRlckhlaWdodDogbnVtYmVyLFxuICBlZmZlY3RpdmVGaWx0ZXJXaWR0aDogbnVtYmVyLFxuICBwYWRJbmZvOiBQYWRJbmZvLFxuICBpblNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgb3V0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICBmaWx0ZXJTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl1cbn07XG5cbi8qKlxuICpcbiAqIEBwYXJhbSBpbnB1dFNoYXBlIElucHV0IHRlbnNvciBzaGFwZSBpcyBvZiB0aGUgZm9sbG93aW5nIGRpbWVuc2lvbnM6XG4gKiAgICAgYFtiYXRjaCwgaGVpZ2h0LCB3aWR0aCwgaW5DaGFubmVsc11gLlxuICogQHBhcmFtIGZpbHRlclNoYXBlIFRoZSBmaWx0ZXIgc2hhcGUgaXMgb2YgdGhlIGZvbGxvd2luZyBkaW1lbnNpb25zOlxuICogICAgIGBbZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aCwgZGVwdGhdYC5cbiAqIEBwYXJhbSBzdHJpZGVzIFRoZSBzdHJpZGVzIG9mIHRoZSBzbGlkaW5nIHdpbmRvdyBmb3IgZWFjaCBkaW1lbnNpb24gb2YgdGhlXG4gKiAgICAgaW5wdXQgdGVuc29yOiBgW3N0cmlkZUhlaWdodCwgc3RyaWRlV2lkdGhdYC5cbiAqICAgICBJZiBgc3RyaWRlc2AgaXMgYSBzaW5nbGUgbnVtYmVyLFxuICogICAgIHRoZW4gYHN0cmlkZUhlaWdodCA9PSBzdHJpZGVXaWR0aGAuXG4gKiBAcGFyYW0gcGFkIFRoZSB0eXBlIG9mIHBhZGRpbmcgYWxnb3JpdGhtLlxuICogICAgLSBgc2FtZWAgYW5kIHN0cmlkZSAxOiBvdXRwdXQgd2lsbCBiZSBvZiBzYW1lIHNpemUgYXMgaW5wdXQsXG4gKiAgICAgICByZWdhcmRsZXNzIG9mIGZpbHRlciBzaXplLlxuICogICAgLSBgdmFsaWRgOiBvdXRwdXQgd2lsbCBiZSBzbWFsbGVyIHRoYW4gaW5wdXQgaWYgZmlsdGVyIGlzIGxhcmdlclxuICogICAgICAgdGhhbiAxKjF4MS5cbiAqICAgIC0gRm9yIG1vcmUgaW5mbywgc2VlIHRoaXMgZ3VpZGU6XG4gKiAgICAgW2h0dHBzOi8vd3d3LnRlbnNvcmZsb3cub3JnL2FwaV9kb2NzL3B5dGhvbi90Zi9ubi9jb252b2x1dGlvbl0oXG4gKiAgICAgICAgICBodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udm9sdXRpb24pXG4gKiBAcGFyYW0gZGF0YUZvcm1hdCBUaGUgZGF0YSBmb3JtYXQgb2YgdGhlIGlucHV0IGFuZCBvdXRwdXQgZGF0YS5cbiAqICAgICBEZWZhdWx0cyB0byAnTkhXQycuXG4gKiBAcGFyYW0gZGlsYXRpb25zIFRoZSBkaWxhdGlvbiByYXRlczogYFtkaWxhdGlvbkhlaWdodCwgZGlsYXRpb25XaWR0aF1gLlxuICogICAgIERlZmF1bHRzIHRvIGBbMSwgMV1gLiBJZiBgZGlsYXRpb25zYCBpcyBhIHNpbmdsZSBudW1iZXIsIHRoZW5cbiAqICAgICBgZGlsYXRpb25IZWlnaHQgPT0gZGlsYXRpb25XaWR0aGAuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlRGlsYXRpb24yREluZm8oXG4gICAgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgZmlsdGVyU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgc3RyaWRlczogbnVtYmVyfFtudW1iZXIsIG51bWJlcl0sXG4gICAgcGFkOiAnc2FtZSd8J3ZhbGlkJ3xudW1iZXIsIGRhdGFGb3JtYXQ6ICdOSFdDJyA9ICdOSFdDJyxcbiAgICBkaWxhdGlvbnM6IG51bWJlcnxbbnVtYmVyLCBudW1iZXJdKSB7XG4gIC8vIGBjb21wdXRlckNvbnYyREluZm9gIHJlcXVpcmUgZmlsdGVyU2hhcGUgdG8gYmUgaW4gdGhlIGRpbWVuc2lvbiBvZjpcbiAgLy8gYFtmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoLCBkZXB0aCwgb3V0RGVwdGhdYCwgZGlsYXRpb24yZCBkb2Vzbid0IGhhdmVcbiAgLy8gb3V0RGVwdGgsIGl0IHNob3VsZCBoYXZlIHRoZSBzYW1lIGRlcHRoIGFzIHRoZSBpbnB1dC5cbiAgLy8gSW5wdXQgc2hhcGU6IFtiYXRjaCwgaGVpZ2h0LCB3aWR0aCwgaW5DaGFubmVsc11cbiAgY29uc3QgaW5wdXRDaGFubmVscyA9IGlucHV0U2hhcGVbM107XG4gIGNvbnN0ICRmaWx0ZXJTaGFwZSA9XG4gICAgICBbLi4uZmlsdGVyU2hhcGUsIGlucHV0Q2hhbm5lbHNdIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBjb25zdCAkZGF0YUZvcm1hdCA9IGNvbnZlcnRDb252MkREYXRhRm9ybWF0KGRhdGFGb3JtYXQpO1xuXG4gIHJldHVybiBjb21wdXRlQ29udjJESW5mbyhcbiAgICAgIGlucHV0U2hhcGUsICRmaWx0ZXJTaGFwZSwgc3RyaWRlcywgZGlsYXRpb25zLCBwYWQsXG4gICAgICBudWxsIC8qIHJvdW5kaW5nTW9kZSAqLywgbnVsbCAvKiBkZXB0aFdpc2UgKi8sICRkYXRhRm9ybWF0KTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVQb29sMkRJbmZvKFxuICAgIGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIGZpbHRlclNpemU6IFtudW1iZXIsIG51bWJlcl18bnVtYmVyLCBzdHJpZGVzOiBudW1iZXJ8W251bWJlciwgbnVtYmVyXSxcbiAgICBkaWxhdGlvbnM6IG51bWJlcnxbbnVtYmVyLCBudW1iZXJdLFxuICAgIHBhZDogJ3NhbWUnfCd2YWxpZCd8bnVtYmVyfEV4cGxpY2l0UGFkZGluZyxcbiAgICByb3VuZGluZ01vZGU/OiAnZmxvb3InfCdyb3VuZCd8J2NlaWwnLFxuICAgIGRhdGFGb3JtYXQ6ICdjaGFubmVsc0ZpcnN0J3wnY2hhbm5lbHNMYXN0JyA9ICdjaGFubmVsc0xhc3QnKTogQ29udjJESW5mbyB7XG4gIGNvbnN0IFtmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoXSA9IHBhcnNlVHVwbGVQYXJhbShmaWx0ZXJTaXplKTtcblxuICBsZXQgZmlsdGVyU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcpIHtcbiAgICBmaWx0ZXJTaGFwZSA9IFtmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoLCBpblNoYXBlWzNdLCBpblNoYXBlWzNdXTtcbiAgfSBlbHNlIGlmIChkYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNGaXJzdCcpIHtcbiAgICBmaWx0ZXJTaGFwZSA9IFtmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoLCBpblNoYXBlWzFdLCBpblNoYXBlWzFdXTtcbiAgfSBlbHNlIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFVua25vd24gZGF0YUZvcm1hdCAke2RhdGFGb3JtYXR9YCk7XG4gIH1cblxuICByZXR1cm4gY29tcHV0ZUNvbnYyREluZm8oXG4gICAgICBpblNoYXBlLCBmaWx0ZXJTaGFwZSwgc3RyaWRlcywgZGlsYXRpb25zLCBwYWQsIHJvdW5kaW5nTW9kZSwgZmFsc2UsXG4gICAgICBkYXRhRm9ybWF0KTtcbn1cblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgaW5mb3JtYXRpb24gZm9yIGEgZm9yd2FyZCBwYXNzIG9mIGEgcG9vbGluZzNEIG9wZXJhdGlvbi5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVQb29sM0RJbmZvKFxuICAgIGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgZmlsdGVyU2l6ZTogbnVtYmVyfFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICBzdHJpZGVzOiBudW1iZXJ8W251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIGRpbGF0aW9uczogbnVtYmVyfFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgcGFkOiAnc2FtZSd8J3ZhbGlkJ3xudW1iZXIsXG4gICAgcm91bmRpbmdNb2RlPzogJ2Zsb29yJ3wncm91bmQnfCdjZWlsJyxcbiAgICBkYXRhRm9ybWF0OiAnTkRIV0MnfCdOQ0RIVycgPSAnTkRIV0MnKTogQ29udjNESW5mbyB7XG4gIGNvbnN0IFtmaWx0ZXJEZXB0aCwgZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aF0gPSBwYXJzZTNUdXBsZVBhcmFtKGZpbHRlclNpemUpO1xuXG4gIGxldCBmaWx0ZXJTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgbGV0ICRkYXRhRm9ybWF0OiAnY2hhbm5lbHNGaXJzdCd8J2NoYW5uZWxzTGFzdCc7XG4gIGlmIChkYXRhRm9ybWF0ID09PSAnTkRIV0MnKSB7XG4gICAgJGRhdGFGb3JtYXQgPSAnY2hhbm5lbHNMYXN0JztcbiAgICBmaWx0ZXJTaGFwZSA9XG4gICAgICAgIFtmaWx0ZXJEZXB0aCwgZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aCwgaW5TaGFwZVs0XSwgaW5TaGFwZVs0XV07XG4gIH0gZWxzZSBpZiAoZGF0YUZvcm1hdCA9PT0gJ05DREhXJykge1xuICAgICRkYXRhRm9ybWF0ID0gJ2NoYW5uZWxzRmlyc3QnO1xuICAgIGZpbHRlclNoYXBlID1cbiAgICAgICAgW2ZpbHRlckRlcHRoLCBmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoLCBpblNoYXBlWzFdLCBpblNoYXBlWzFdXTtcbiAgfSBlbHNlIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFVua25vd24gZGF0YUZvcm1hdCAke2RhdGFGb3JtYXR9YCk7XG4gIH1cblxuICByZXR1cm4gY29tcHV0ZUNvbnYzREluZm8oXG4gICAgICBpblNoYXBlLCBmaWx0ZXJTaGFwZSwgc3RyaWRlcywgZGlsYXRpb25zLCBwYWQsIGZhbHNlLCAkZGF0YUZvcm1hdCxcbiAgICAgIHJvdW5kaW5nTW9kZSk7XG59XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIGluZm9ybWF0aW9uIGZvciBhIGZvcndhcmQgcGFzcyBvZiBhIGNvbnZvbHV0aW9uL3Bvb2xpbmdcbiAqIG9wZXJhdGlvbi5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVDb252MkRJbmZvKFxuICAgIGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIGZpbHRlclNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICBzdHJpZGVzOiBudW1iZXJ8W251bWJlciwgbnVtYmVyXSwgZGlsYXRpb25zOiBudW1iZXJ8W251bWJlciwgbnVtYmVyXSxcbiAgICBwYWQ6ICdzYW1lJ3wndmFsaWQnfG51bWJlcnxFeHBsaWNpdFBhZGRpbmcsXG4gICAgcm91bmRpbmdNb2RlPzogJ2Zsb29yJ3wncm91bmQnfCdjZWlsJywgZGVwdGh3aXNlID0gZmFsc2UsXG4gICAgZGF0YUZvcm1hdDogJ2NoYW5uZWxzRmlyc3QnfCdjaGFubmVsc0xhc3QnID0gJ2NoYW5uZWxzTGFzdCcpOiBDb252MkRJbmZvIHtcbiAgbGV0IFtiYXRjaFNpemUsIGluSGVpZ2h0LCBpbldpZHRoLCBpbkNoYW5uZWxzXSA9IFstMSwgLTEsIC0xLCAtMV07XG4gIGlmIChkYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNMYXN0Jykge1xuICAgIFtiYXRjaFNpemUsIGluSGVpZ2h0LCBpbldpZHRoLCBpbkNoYW5uZWxzXSA9IGluU2hhcGU7XG4gIH0gZWxzZSBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgW2JhdGNoU2l6ZSwgaW5DaGFubmVscywgaW5IZWlnaHQsIGluV2lkdGhdID0gaW5TaGFwZTtcbiAgfSBlbHNlIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFVua25vd24gZGF0YUZvcm1hdCAke2RhdGFGb3JtYXR9YCk7XG4gIH1cblxuICBjb25zdCBbZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aCwgLCBmaWx0ZXJDaGFubmVsc10gPSBmaWx0ZXJTaGFwZTtcbiAgY29uc3QgW3N0cmlkZUhlaWdodCwgc3RyaWRlV2lkdGhdID0gcGFyc2VUdXBsZVBhcmFtKHN0cmlkZXMpO1xuICBjb25zdCBbZGlsYXRpb25IZWlnaHQsIGRpbGF0aW9uV2lkdGhdID0gcGFyc2VUdXBsZVBhcmFtKGRpbGF0aW9ucyk7XG5cbiAgY29uc3QgZWZmZWN0aXZlRmlsdGVySGVpZ2h0ID1cbiAgICAgIGdldEVmZmVjdGl2ZUZpbHRlclNpemUoZmlsdGVySGVpZ2h0LCBkaWxhdGlvbkhlaWdodCk7XG4gIGNvbnN0IGVmZmVjdGl2ZUZpbHRlcldpZHRoID1cbiAgICAgIGdldEVmZmVjdGl2ZUZpbHRlclNpemUoZmlsdGVyV2lkdGgsIGRpbGF0aW9uV2lkdGgpO1xuICBjb25zdCB7cGFkSW5mbywgb3V0SGVpZ2h0LCBvdXRXaWR0aH0gPSBnZXRQYWRBbmRPdXRJbmZvKFxuICAgICAgcGFkLCBpbkhlaWdodCwgaW5XaWR0aCwgc3RyaWRlSGVpZ2h0LCBzdHJpZGVXaWR0aCwgZWZmZWN0aXZlRmlsdGVySGVpZ2h0LFxuICAgICAgZWZmZWN0aXZlRmlsdGVyV2lkdGgsIHJvdW5kaW5nTW9kZSwgZGF0YUZvcm1hdCk7XG5cbiAgY29uc3Qgb3V0Q2hhbm5lbHMgPSBkZXB0aHdpc2UgPyBmaWx0ZXJDaGFubmVscyAqIGluQ2hhbm5lbHMgOiBmaWx0ZXJDaGFubmVscztcblxuICBsZXQgb3V0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzRmlyc3QnKSB7XG4gICAgb3V0U2hhcGUgPSBbYmF0Y2hTaXplLCBvdXRDaGFubmVscywgb3V0SGVpZ2h0LCBvdXRXaWR0aF07XG4gIH0gZWxzZSBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcpIHtcbiAgICBvdXRTaGFwZSA9IFtiYXRjaFNpemUsIG91dEhlaWdodCwgb3V0V2lkdGgsIG91dENoYW5uZWxzXTtcbiAgfVxuXG4gIHJldHVybiB7XG4gICAgYmF0Y2hTaXplLFxuICAgIGRhdGFGb3JtYXQsXG4gICAgaW5IZWlnaHQsXG4gICAgaW5XaWR0aCxcbiAgICBpbkNoYW5uZWxzLFxuICAgIG91dEhlaWdodCxcbiAgICBvdXRXaWR0aCxcbiAgICBvdXRDaGFubmVscyxcbiAgICBwYWRJbmZvLFxuICAgIHN0cmlkZUhlaWdodCxcbiAgICBzdHJpZGVXaWR0aCxcbiAgICBmaWx0ZXJIZWlnaHQsXG4gICAgZmlsdGVyV2lkdGgsXG4gICAgZWZmZWN0aXZlRmlsdGVySGVpZ2h0LFxuICAgIGVmZmVjdGl2ZUZpbHRlcldpZHRoLFxuICAgIGRpbGF0aW9uSGVpZ2h0LFxuICAgIGRpbGF0aW9uV2lkdGgsXG4gICAgaW5TaGFwZSxcbiAgICBvdXRTaGFwZSxcbiAgICBmaWx0ZXJTaGFwZVxuICB9O1xufVxuXG4vKipcbiAqIEluZm9ybWF0aW9uIGFib3V0IHRoZSBmb3J3YXJkIHBhc3Mgb2YgYSAzRCBjb252b2x1dGlvbi9wb29saW5nIG9wZXJhdGlvbi5cbiAqIEl0IGluY2x1ZGVzIGlucHV0IGFuZCBvdXRwdXQgc2hhcGUsIHN0cmlkZXMsIGZpbHRlciBzaXplIGFuZCBwYWRkaW5nXG4gKiBpbmZvcm1hdGlvbi5cbiAqL1xuZXhwb3J0IHR5cGUgQ29udjNESW5mbyA9IHtcbiAgYmF0Y2hTaXplOiBudW1iZXIsXG4gIGluRGVwdGg6IG51bWJlcixcbiAgaW5IZWlnaHQ6IG51bWJlcixcbiAgaW5XaWR0aDogbnVtYmVyLFxuICBpbkNoYW5uZWxzOiBudW1iZXIsXG4gIG91dERlcHRoOiBudW1iZXIsXG4gIG91dEhlaWdodDogbnVtYmVyLFxuICBvdXRXaWR0aDogbnVtYmVyLFxuICBvdXRDaGFubmVsczogbnVtYmVyLFxuICBkYXRhRm9ybWF0OiAnY2hhbm5lbHNGaXJzdCd8J2NoYW5uZWxzTGFzdCcsXG4gIHN0cmlkZURlcHRoOiBudW1iZXIsXG4gIHN0cmlkZUhlaWdodDogbnVtYmVyLFxuICBzdHJpZGVXaWR0aDogbnVtYmVyLFxuICBkaWxhdGlvbkRlcHRoOiBudW1iZXIsXG4gIGRpbGF0aW9uSGVpZ2h0OiBudW1iZXIsXG4gIGRpbGF0aW9uV2lkdGg6IG51bWJlcixcbiAgZmlsdGVyRGVwdGg6IG51bWJlcixcbiAgZmlsdGVySGVpZ2h0OiBudW1iZXIsXG4gIGZpbHRlcldpZHRoOiBudW1iZXIsXG4gIGVmZmVjdGl2ZUZpbHRlckRlcHRoOiBudW1iZXIsXG4gIGVmZmVjdGl2ZUZpbHRlckhlaWdodDogbnVtYmVyLFxuICBlZmZlY3RpdmVGaWx0ZXJXaWR0aDogbnVtYmVyLFxuICBwYWRJbmZvOiBQYWRJbmZvM0QsXG4gIGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gIG91dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICBmaWx0ZXJTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXVxufTtcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgaW5mb3JtYXRpb24gZm9yIGEgZm9yd2FyZCBwYXNzIG9mIGEgM0QgY29udm9sdXRpb24vcG9vbGluZ1xuICogb3BlcmF0aW9uLlxuICovXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZUNvbnYzREluZm8oXG4gICAgaW5TaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICBmaWx0ZXJTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICBzdHJpZGVzOiBudW1iZXJ8W251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIGRpbGF0aW9uczogbnVtYmVyfFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgcGFkOiAnc2FtZSd8J3ZhbGlkJ3xudW1iZXIsXG4gICAgZGVwdGh3aXNlID0gZmFsc2UsXG4gICAgZGF0YUZvcm1hdDogJ2NoYW5uZWxzRmlyc3QnfCdjaGFubmVsc0xhc3QnID0gJ2NoYW5uZWxzTGFzdCcsXG4gICAgcm91bmRpbmdNb2RlPzogJ2Zsb29yJ3wncm91bmQnfCdjZWlsJyk6IENvbnYzREluZm8ge1xuICBsZXQgW2JhdGNoU2l6ZSwgaW5EZXB0aCwgaW5IZWlnaHQsIGluV2lkdGgsIGluQ2hhbm5lbHNdID1cbiAgICAgIFstMSwgLTEsIC0xLCAtMSwgLTFdO1xuICBpZiAoZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcpIHtcbiAgICBbYmF0Y2hTaXplLCBpbkRlcHRoLCBpbkhlaWdodCwgaW5XaWR0aCwgaW5DaGFubmVsc10gPSBpblNoYXBlO1xuICB9IGVsc2UgaWYgKGRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0Jykge1xuICAgIFtiYXRjaFNpemUsIGluQ2hhbm5lbHMsIGluRGVwdGgsIGluSGVpZ2h0LCBpbldpZHRoXSA9IGluU2hhcGU7XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBVbmtub3duIGRhdGFGb3JtYXQgJHtkYXRhRm9ybWF0fWApO1xuICB9XG5cbiAgY29uc3QgW2ZpbHRlckRlcHRoLCBmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoLCAsIGZpbHRlckNoYW5uZWxzXSA9XG4gICAgICBmaWx0ZXJTaGFwZTtcbiAgY29uc3QgW3N0cmlkZURlcHRoLCBzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoXSA9IHBhcnNlM1R1cGxlUGFyYW0oc3RyaWRlcyk7XG4gIGNvbnN0IFtkaWxhdGlvbkRlcHRoLCBkaWxhdGlvbkhlaWdodCwgZGlsYXRpb25XaWR0aF0gPVxuICAgICAgcGFyc2UzVHVwbGVQYXJhbShkaWxhdGlvbnMpO1xuXG4gIGNvbnN0IGVmZmVjdGl2ZUZpbHRlckRlcHRoID1cbiAgICAgIGdldEVmZmVjdGl2ZUZpbHRlclNpemUoZmlsdGVyRGVwdGgsIGRpbGF0aW9uRGVwdGgpO1xuICBjb25zdCBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgPVxuICAgICAgZ2V0RWZmZWN0aXZlRmlsdGVyU2l6ZShmaWx0ZXJIZWlnaHQsIGRpbGF0aW9uSGVpZ2h0KTtcbiAgY29uc3QgZWZmZWN0aXZlRmlsdGVyV2lkdGggPVxuICAgICAgZ2V0RWZmZWN0aXZlRmlsdGVyU2l6ZShmaWx0ZXJXaWR0aCwgZGlsYXRpb25XaWR0aCk7XG4gIGNvbnN0IHtwYWRJbmZvLCBvdXREZXB0aCwgb3V0SGVpZ2h0LCBvdXRXaWR0aH0gPSBnZXQzRFBhZEFuZE91dEluZm8oXG4gICAgICBwYWQsIGluRGVwdGgsIGluSGVpZ2h0LCBpbldpZHRoLCBzdHJpZGVEZXB0aCwgc3RyaWRlSGVpZ2h0LCBzdHJpZGVXaWR0aCxcbiAgICAgIGVmZmVjdGl2ZUZpbHRlckRlcHRoLCBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQsIGVmZmVjdGl2ZUZpbHRlcldpZHRoLFxuICAgICAgcm91bmRpbmdNb2RlKTtcblxuICBjb25zdCBvdXRDaGFubmVscyA9IGRlcHRod2lzZSA/IGZpbHRlckNoYW5uZWxzICogaW5DaGFubmVscyA6IGZpbHRlckNoYW5uZWxzO1xuXG4gIGxldCBvdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgaWYgKGRhdGFGb3JtYXQgPT09ICdjaGFubmVsc0ZpcnN0Jykge1xuICAgIG91dFNoYXBlID0gW2JhdGNoU2l6ZSwgb3V0Q2hhbm5lbHMsIG91dERlcHRoLCBvdXRIZWlnaHQsIG91dFdpZHRoXTtcbiAgfSBlbHNlIGlmIChkYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNMYXN0Jykge1xuICAgIG91dFNoYXBlID0gW2JhdGNoU2l6ZSwgb3V0RGVwdGgsIG91dEhlaWdodCwgb3V0V2lkdGgsIG91dENoYW5uZWxzXTtcbiAgfVxuXG4gIHJldHVybiB7XG4gICAgYmF0Y2hTaXplLFxuICAgIGRhdGFGb3JtYXQsXG4gICAgaW5EZXB0aCxcbiAgICBpbkhlaWdodCxcbiAgICBpbldpZHRoLFxuICAgIGluQ2hhbm5lbHMsXG4gICAgb3V0RGVwdGgsXG4gICAgb3V0SGVpZ2h0LFxuICAgIG91dFdpZHRoLFxuICAgIG91dENoYW5uZWxzLFxuICAgIHBhZEluZm8sXG4gICAgc3RyaWRlRGVwdGgsXG4gICAgc3RyaWRlSGVpZ2h0LFxuICAgIHN0cmlkZVdpZHRoLFxuICAgIGZpbHRlckRlcHRoLFxuICAgIGZpbHRlckhlaWdodCxcbiAgICBmaWx0ZXJXaWR0aCxcbiAgICBlZmZlY3RpdmVGaWx0ZXJEZXB0aCxcbiAgICBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQsXG4gICAgZWZmZWN0aXZlRmlsdGVyV2lkdGgsXG4gICAgZGlsYXRpb25EZXB0aCxcbiAgICBkaWxhdGlvbkhlaWdodCxcbiAgICBkaWxhdGlvbldpZHRoLFxuICAgIGluU2hhcGUsXG4gICAgb3V0U2hhcGUsXG4gICAgZmlsdGVyU2hhcGVcbiAgfTtcbn1cblxuZnVuY3Rpb24gY29tcHV0ZU91dHB1dFNoYXBlMkQoXG4gICAgaW5TaGFwZTogW251bWJlciwgbnVtYmVyXSwgZmllbGRTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLFxuICAgIHplcm9QYWQ/OiBudW1iZXIsIHJvdW5kaW5nTW9kZT86ICdmbG9vcid8J3JvdW5kJ3wnY2VpbCcpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgaWYgKHplcm9QYWQgPT0gbnVsbCkge1xuICAgIHplcm9QYWQgPSBjb21wdXRlRGVmYXVsdFBhZChpblNoYXBlLCBmaWVsZFNpemUsIHN0cmlkZSk7XG4gIH1cbiAgY29uc3QgaW5wdXRSb3dzID0gaW5TaGFwZVswXTtcbiAgY29uc3QgaW5wdXRDb2xzID0gaW5TaGFwZVsxXTtcblxuICBjb25zdCBvdXRwdXRSb3dzID1cbiAgICAgIHJvdW5kKChpbnB1dFJvd3MgLSBmaWVsZFNpemUgKyAyICogemVyb1BhZCkgLyBzdHJpZGUgKyAxLCByb3VuZGluZ01vZGUpO1xuICBjb25zdCBvdXRwdXRDb2xzID1cbiAgICAgIHJvdW5kKChpbnB1dENvbHMgLSBmaWVsZFNpemUgKyAyICogemVyb1BhZCkgLyBzdHJpZGUgKyAxLCByb3VuZGluZ01vZGUpO1xuXG4gIHJldHVybiBbb3V0cHV0Um93cywgb3V0cHV0Q29sc107XG59XG5cbmZ1bmN0aW9uIGNvbXB1dGVPdXRwdXRTaGFwZTREKFxuICAgIGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIGZpbHRlclNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIG91dENoYW5uZWxzOiBudW1iZXIsXG4gICAgc3RyaWRlczogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCB6ZXJvUGFkPzogbnVtYmVyLFxuICAgIHJvdW5kaW5nTW9kZT86ICdmbG9vcid8J3JvdW5kJ3wnY2VpbCcpOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSB7XG4gIGlmICh6ZXJvUGFkID09IG51bGwpIHtcbiAgICB6ZXJvUGFkID0gY29tcHV0ZURlZmF1bHRQYWQoaW5TaGFwZSwgZmlsdGVyU2hhcGVbMF0sIHN0cmlkZXNbMF0pO1xuICB9XG4gIGNvbnN0IG91dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFswLCAwLCAwLCBvdXRDaGFubmVsc107XG4gIGZvciAobGV0IGluZGV4ID0gMDsgaW5kZXggPCAzOyBpbmRleCsrKSB7XG4gICAgaWYgKGluU2hhcGVbaW5kZXhdICsgMiAqIHplcm9QYWQgPj0gZmlsdGVyU2hhcGVbaW5kZXhdKSB7XG4gICAgICBvdXRTaGFwZVtpbmRleF0gPSByb3VuZChcbiAgICAgICAgICAoaW5TaGFwZVtpbmRleF0gLSBmaWx0ZXJTaGFwZVtpbmRleF0gKyAyICogemVyb1BhZCkgLyBzdHJpZGVzW2luZGV4XSArXG4gICAgICAgICAgICAgIDEsXG4gICAgICAgICAgcm91bmRpbmdNb2RlKTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIG91dFNoYXBlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZURlZmF1bHRQYWQoXG4gICAgaW5wdXRTaGFwZTogW251bWJlciwgbnVtYmVyXXxbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICBmaWVsZFNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIGRpbGF0aW9uID0gMSk6IG51bWJlciB7XG4gIGNvbnN0IGVmZmVjdGl2ZUZpZWxkU2l6ZSA9IGdldEVmZmVjdGl2ZUZpbHRlclNpemUoZmllbGRTaXplLCBkaWxhdGlvbik7XG4gIHJldHVybiBNYXRoLmZsb29yKFxuICAgICAgKGlucHV0U2hhcGVbMF0gKiAoc3RyaWRlIC0gMSkgLSBzdHJpZGUgKyBlZmZlY3RpdmVGaWVsZFNpemUpIC8gMik7XG59XG5cbmZ1bmN0aW9uIHBhcnNlVHVwbGVQYXJhbShwYXJhbTogbnVtYmVyfG51bWJlcltdKTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdIHtcbiAgaWYgKHR5cGVvZiBwYXJhbSA9PT0gJ251bWJlcicpIHtcbiAgICByZXR1cm4gW3BhcmFtLCBwYXJhbSwgcGFyYW1dO1xuICB9XG4gIGlmIChwYXJhbS5sZW5ndGggPT09IDIpIHtcbiAgICByZXR1cm4gW3BhcmFtWzBdLCBwYXJhbVsxXSwgMV07XG4gIH1cbiAgcmV0dXJuIHBhcmFtIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbn1cblxuZnVuY3Rpb24gcGFyc2UzVHVwbGVQYXJhbShwYXJhbTogbnVtYmVyfFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSk6XG4gICAgW251bWJlciwgbnVtYmVyLCBudW1iZXJdIHtcbiAgcmV0dXJuIHR5cGVvZiBwYXJhbSA9PT0gJ251bWJlcicgPyBbcGFyYW0sIHBhcmFtLCBwYXJhbV0gOiBwYXJhbTtcbn1cblxuLyogU2VlIGh0dHBzOi8vd3d3LnRlbnNvcmZsb3cub3JnL2FwaV9kb2NzL3B5dGhvbi90Zi9ubi9hdHJvdXNfY29udjJkXG4gKiBBdHJvdXMgY29udm9sdXRpb24gaXMgZXF1aXZhbGVudCB0byBzdGFuZGFyZCBjb252b2x1dGlvbiB3aXRoIHVwc2FtcGxlZFxuICogZmlsdGVycyB3aXRoIGVmZmVjdGl2ZV9maWx0ZXJfaGVpZ2h0ID1cbiAqIGZpbHRlcl9oZWlnaHQgKyAoZmlsdGVyX2hlaWdodCAtIDEpICogKGRpbGF0aW9uIC0gMSlcbiAqIGFuZCBlZmZlY3RpdmVfZmlsdGVyX3dpZHRoID1cbiAqIGZpbHRlcl93aWR0aCArIChmaWx0ZXJfd2lkdGggLSAxKSAqIChkaWxhdGlvbiAtIDEpLFxuICogcHJvZHVjZWQgYnkgaW5zZXJ0aW5nIGRpbGF0aW9uIC0gMSB6ZXJvcyBhbG9uZyBjb25zZWN1dGl2ZSBlbGVtZW50cyBhY3Jvc3NcbiAqIHRoZSBmaWx0ZXJzJyBzcGF0aWFsIGRpbWVuc2lvbnMuXG4gKiBXaGVuIHRoZXJlIGlzIGEgZGlsYXRpb24sIHRoaXMgY29udmVydHMgYSBmaWx0ZXIgZGltZW5zaW9uIHRvIHRoZVxuICogZWZmZWN0aXZlIGZpbHRlciBkaW1lbnNpb24sIHNvIGl0IGNhbiBiZSB1c2VkIGluIGEgc3RhbmRhcmQgY29udm9sdXRpb24uXG4gKi9cbmZ1bmN0aW9uIGdldEVmZmVjdGl2ZUZpbHRlclNpemUoZmlsdGVyU2l6ZTogbnVtYmVyLCBkaWxhdGlvbjogbnVtYmVyKSB7XG4gIGlmIChkaWxhdGlvbiA8PSAxKSB7XG4gICAgcmV0dXJuIGZpbHRlclNpemU7XG4gIH1cblxuICByZXR1cm4gZmlsdGVyU2l6ZSArIChmaWx0ZXJTaXplIC0gMSkgKiAoZGlsYXRpb24gLSAxKTtcbn1cblxuZnVuY3Rpb24gZ2V0UGFkQW5kT3V0SW5mbyhcbiAgICBwYWQ6ICdzYW1lJ3wndmFsaWQnfG51bWJlcnxFeHBsaWNpdFBhZGRpbmcsIGluSGVpZ2h0OiBudW1iZXIsXG4gICAgaW5XaWR0aDogbnVtYmVyLCBzdHJpZGVIZWlnaHQ6IG51bWJlciwgc3RyaWRlV2lkdGg6IG51bWJlcixcbiAgICBmaWx0ZXJIZWlnaHQ6IG51bWJlciwgZmlsdGVyV2lkdGg6IG51bWJlcixcbiAgICByb3VuZGluZ01vZGU6ICdmbG9vcid8J3JvdW5kJ3wnY2VpbCcsXG4gICAgZGF0YUZvcm1hdDogJ2NoYW5uZWxzRmlyc3QnfFxuICAgICdjaGFubmVsc0xhc3QnKToge3BhZEluZm86IFBhZEluZm8sIG91dEhlaWdodDogbnVtYmVyLCBvdXRXaWR0aDogbnVtYmVyfSB7XG4gIGxldCBwYWRJbmZvOiBQYWRJbmZvO1xuICBsZXQgb3V0SGVpZ2h0OiBudW1iZXI7XG4gIGxldCBvdXRXaWR0aDogbnVtYmVyO1xuXG4gIGlmICh0eXBlb2YgcGFkID09PSAnbnVtYmVyJykge1xuICAgIGNvbnN0IHBhZFR5cGUgPSAocGFkID09PSAwKSA/ICdWQUxJRCcgOiAnTlVNQkVSJztcbiAgICBwYWRJbmZvID0ge3RvcDogcGFkLCBib3R0b206IHBhZCwgbGVmdDogcGFkLCByaWdodDogcGFkLCB0eXBlOiBwYWRUeXBlfTtcbiAgICBjb25zdCBvdXRTaGFwZSA9IGNvbXB1dGVPdXRwdXRTaGFwZTJEKFxuICAgICAgICBbaW5IZWlnaHQsIGluV2lkdGhdLCBmaWx0ZXJIZWlnaHQsIHN0cmlkZUhlaWdodCwgcGFkLCByb3VuZGluZ01vZGUpO1xuICAgIG91dEhlaWdodCA9IG91dFNoYXBlWzBdO1xuICAgIG91dFdpZHRoID0gb3V0U2hhcGVbMV07XG4gIH0gZWxzZSBpZiAocGFkID09PSAnc2FtZScpIHtcbiAgICBvdXRIZWlnaHQgPSBNYXRoLmNlaWwoaW5IZWlnaHQgLyBzdHJpZGVIZWlnaHQpO1xuICAgIG91dFdpZHRoID0gTWF0aC5jZWlsKGluV2lkdGggLyBzdHJpZGVXaWR0aCk7XG4gICAgY29uc3QgcGFkQWxvbmdIZWlnaHQgPVxuICAgICAgICBNYXRoLm1heCgwLCAob3V0SGVpZ2h0IC0gMSkgKiBzdHJpZGVIZWlnaHQgKyBmaWx0ZXJIZWlnaHQgLSBpbkhlaWdodCk7XG4gICAgY29uc3QgcGFkQWxvbmdXaWR0aCA9XG4gICAgICAgIE1hdGgubWF4KDAsIChvdXRXaWR0aCAtIDEpICogc3RyaWRlV2lkdGggKyBmaWx0ZXJXaWR0aCAtIGluV2lkdGgpO1xuICAgIGNvbnN0IHRvcCA9IE1hdGguZmxvb3IocGFkQWxvbmdIZWlnaHQgLyAyKTtcbiAgICBjb25zdCBib3R0b20gPSBwYWRBbG9uZ0hlaWdodCAtIHRvcDtcbiAgICBjb25zdCBsZWZ0ID0gTWF0aC5mbG9vcihwYWRBbG9uZ1dpZHRoIC8gMik7XG4gICAgY29uc3QgcmlnaHQgPSBwYWRBbG9uZ1dpZHRoIC0gbGVmdDtcbiAgICBwYWRJbmZvID0ge3RvcCwgYm90dG9tLCBsZWZ0LCByaWdodCwgdHlwZTogJ1NBTUUnfTtcbiAgfSBlbHNlIGlmIChwYWQgPT09ICd2YWxpZCcpIHtcbiAgICBwYWRJbmZvID0ge3RvcDogMCwgYm90dG9tOiAwLCBsZWZ0OiAwLCByaWdodDogMCwgdHlwZTogJ1ZBTElEJ307XG4gICAgb3V0SGVpZ2h0ID0gTWF0aC5jZWlsKChpbkhlaWdodCAtIGZpbHRlckhlaWdodCArIDEpIC8gc3RyaWRlSGVpZ2h0KTtcbiAgICBvdXRXaWR0aCA9IE1hdGguY2VpbCgoaW5XaWR0aCAtIGZpbHRlcldpZHRoICsgMSkgLyBzdHJpZGVXaWR0aCk7XG4gIH0gZWxzZSBpZiAodHlwZW9mIHBhZCA9PT0gJ29iamVjdCcpIHtcbiAgICBjb25zdCB0b3AgPSBkYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNMYXN0JyA/IHBhZFsxXVswXSA6IHBhZFsyXVswXTtcbiAgICBjb25zdCBib3R0b20gPSBkYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNMYXN0JyA/IHBhZFsxXVsxXSA6IHBhZFsyXVsxXTtcbiAgICBjb25zdCBsZWZ0ID0gZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCcgPyBwYWRbMl1bMF0gOiBwYWRbM11bMF07XG4gICAgY29uc3QgcmlnaHQgPSBkYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNMYXN0JyA/IHBhZFsyXVsxXSA6IHBhZFszXVsxXTtcbiAgICBjb25zdCBwYWRUeXBlID0gKHRvcCA9PT0gMCAmJiBib3R0b20gPT09IDAgJiYgbGVmdCA9PT0gMCAmJiByaWdodCA9PT0gMCkgP1xuICAgICAgICAnVkFMSUQnIDpcbiAgICAgICAgJ0VYUExJQ0lUJztcbiAgICBwYWRJbmZvID0ge3RvcCwgYm90dG9tLCBsZWZ0LCByaWdodCwgdHlwZTogcGFkVHlwZX07XG4gICAgb3V0SGVpZ2h0ID0gcm91bmQoXG4gICAgICAgIChpbkhlaWdodCAtIGZpbHRlckhlaWdodCArIHRvcCArIGJvdHRvbSkgLyBzdHJpZGVIZWlnaHQgKyAxLFxuICAgICAgICByb3VuZGluZ01vZGUpO1xuICAgIG91dFdpZHRoID0gcm91bmQoXG4gICAgICAgIChpbldpZHRoIC0gZmlsdGVyV2lkdGggKyBsZWZ0ICsgcmlnaHQpIC8gc3RyaWRlV2lkdGggKyAxLCByb3VuZGluZ01vZGUpO1xuICB9IGVsc2Uge1xuICAgIHRocm93IEVycm9yKGBVbmtub3duIHBhZGRpbmcgcGFyYW1ldGVyOiAke3BhZH1gKTtcbiAgfVxuICByZXR1cm4ge3BhZEluZm8sIG91dEhlaWdodCwgb3V0V2lkdGh9O1xufVxuXG5mdW5jdGlvbiBnZXQzRFBhZEFuZE91dEluZm8oXG4gICAgcGFkOiAnc2FtZSd8J3ZhbGlkJ3xudW1iZXIsIGluRGVwdGg6IG51bWJlciwgaW5IZWlnaHQ6IG51bWJlcixcbiAgICBpbldpZHRoOiBudW1iZXIsIHN0cmlkZURlcHRoOiBudW1iZXIsIHN0cmlkZUhlaWdodDogbnVtYmVyLFxuICAgIHN0cmlkZVdpZHRoOiBudW1iZXIsIGZpbHRlckRlcHRoOiBudW1iZXIsIGZpbHRlckhlaWdodDogbnVtYmVyLFxuICAgIGZpbHRlcldpZHRoOiBudW1iZXIsIHJvdW5kaW5nTW9kZT86ICdmbG9vcid8J3JvdW5kJ3wnY2VpbCcpOiB7XG4gIHBhZEluZm86IFBhZEluZm8zRCxcbiAgb3V0RGVwdGg6IG51bWJlcixcbiAgb3V0SGVpZ2h0OiBudW1iZXIsXG4gIG91dFdpZHRoOiBudW1iZXJcbn0ge1xuICBsZXQgcGFkSW5mbzogUGFkSW5mbzNEO1xuICBsZXQgb3V0RGVwdGg6IG51bWJlcjtcbiAgbGV0IG91dEhlaWdodDogbnVtYmVyO1xuICBsZXQgb3V0V2lkdGg6IG51bWJlcjtcblxuICBpZiAocGFkID09PSAndmFsaWQnKSB7XG4gICAgcGFkID0gMDtcbiAgfVxuXG4gIGlmICh0eXBlb2YgcGFkID09PSAnbnVtYmVyJykge1xuICAgIGNvbnN0IHBhZFR5cGUgPSAocGFkID09PSAwKSA/ICdWQUxJRCcgOiAnTlVNQkVSJztcbiAgICBwYWRJbmZvID0ge1xuICAgICAgdG9wOiBwYWQsXG4gICAgICBib3R0b206IHBhZCxcbiAgICAgIGxlZnQ6IHBhZCxcbiAgICAgIHJpZ2h0OiBwYWQsXG4gICAgICBmcm9udDogcGFkLFxuICAgICAgYmFjazogcGFkLFxuICAgICAgdHlwZTogcGFkVHlwZVxuICAgIH07XG4gICAgY29uc3Qgb3V0U2hhcGUgPSBjb21wdXRlT3V0cHV0U2hhcGU0RChcbiAgICAgICAgW2luRGVwdGgsIGluSGVpZ2h0LCBpbldpZHRoLCAxXSxcbiAgICAgICAgW2ZpbHRlckRlcHRoLCBmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoXSwgMSxcbiAgICAgICAgW3N0cmlkZURlcHRoLCBzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoXSwgcGFkLCByb3VuZGluZ01vZGUpO1xuICAgIG91dERlcHRoID0gb3V0U2hhcGVbMF07XG4gICAgb3V0SGVpZ2h0ID0gb3V0U2hhcGVbMV07XG4gICAgb3V0V2lkdGggPSBvdXRTaGFwZVsyXTtcbiAgfSBlbHNlIGlmIChwYWQgPT09ICdzYW1lJykge1xuICAgIG91dERlcHRoID0gTWF0aC5jZWlsKGluRGVwdGggLyBzdHJpZGVEZXB0aCk7XG4gICAgb3V0SGVpZ2h0ID0gTWF0aC5jZWlsKGluSGVpZ2h0IC8gc3RyaWRlSGVpZ2h0KTtcbiAgICBvdXRXaWR0aCA9IE1hdGguY2VpbChpbldpZHRoIC8gc3RyaWRlV2lkdGgpO1xuICAgIGNvbnN0IHBhZEFsb25nRGVwdGggPSAob3V0RGVwdGggLSAxKSAqIHN0cmlkZURlcHRoICsgZmlsdGVyRGVwdGggLSBpbkRlcHRoO1xuICAgIGNvbnN0IHBhZEFsb25nSGVpZ2h0ID1cbiAgICAgICAgKG91dEhlaWdodCAtIDEpICogc3RyaWRlSGVpZ2h0ICsgZmlsdGVySGVpZ2h0IC0gaW5IZWlnaHQ7XG4gICAgY29uc3QgcGFkQWxvbmdXaWR0aCA9IChvdXRXaWR0aCAtIDEpICogc3RyaWRlV2lkdGggKyBmaWx0ZXJXaWR0aCAtIGluV2lkdGg7XG4gICAgY29uc3QgZnJvbnQgPSBNYXRoLmZsb29yKHBhZEFsb25nRGVwdGggLyAyKTtcbiAgICBjb25zdCBiYWNrID0gcGFkQWxvbmdEZXB0aCAtIGZyb250O1xuICAgIGNvbnN0IHRvcCA9IE1hdGguZmxvb3IocGFkQWxvbmdIZWlnaHQgLyAyKTtcbiAgICBjb25zdCBib3R0b20gPSBwYWRBbG9uZ0hlaWdodCAtIHRvcDtcbiAgICBjb25zdCBsZWZ0ID0gTWF0aC5mbG9vcihwYWRBbG9uZ1dpZHRoIC8gMik7XG4gICAgY29uc3QgcmlnaHQgPSBwYWRBbG9uZ1dpZHRoIC0gbGVmdDtcblxuICAgIHBhZEluZm8gPSB7dG9wLCBib3R0b20sIGxlZnQsIHJpZ2h0LCBmcm9udCwgYmFjaywgdHlwZTogJ1NBTUUnfTtcbiAgfSBlbHNlIHtcbiAgICB0aHJvdyBFcnJvcihgVW5rbm93biBwYWRkaW5nIHBhcmFtZXRlcjogJHtwYWR9YCk7XG4gIH1cbiAgcmV0dXJuIHtwYWRJbmZvLCBvdXREZXB0aCwgb3V0SGVpZ2h0LCBvdXRXaWR0aH07XG59XG5cbi8qKlxuICogUm91bmRzIGEgdmFsdWUgZGVwZW5kaW5nIG9uIHRoZSByb3VuZGluZyBtb2RlXG4gKiBAcGFyYW0gdmFsdWVcbiAqIEBwYXJhbSByb3VuZGluZ01vZGUgQSBzdHJpbmcgZnJvbTogJ2NlaWwnLCAncm91bmQnLCAnZmxvb3InLiBJZiBub25lIGlzXG4gKiAgICAgcHJvdmlkZWQsIGl0IHdpbGwgZGVmYXVsdCB0byB0cnVuY2F0ZS5cbiAqL1xuZnVuY3Rpb24gcm91bmQodmFsdWU6IG51bWJlciwgcm91bmRpbmdNb2RlPzogJ2Zsb29yJ3wncm91bmQnfCdjZWlsJykge1xuICBpZiAoIXJvdW5kaW5nTW9kZSkge1xuICAgIHJldHVybiBNYXRoLnRydW5jKHZhbHVlKTtcbiAgfVxuICBzd2l0Y2ggKHJvdW5kaW5nTW9kZSkge1xuICAgIGNhc2UgJ3JvdW5kJzpcbiAgICAgIC8vIHVzZWQgZm9yIENhZmZlIENvbnZcbiAgICAgIHJldHVybiBNYXRoLnJvdW5kKHZhbHVlKTtcbiAgICBjYXNlICdjZWlsJzpcbiAgICAgIC8vIHVzZWQgZm9yIENhZmZlIFBvb2xcbiAgICAgIHJldHVybiBNYXRoLmNlaWwodmFsdWUpO1xuICAgIGNhc2UgJ2Zsb29yJzpcbiAgICAgIHJldHVybiBNYXRoLmZsb29yKHZhbHVlKTtcbiAgICBkZWZhdWx0OlxuICAgICAgdGhyb3cgbmV3IEVycm9yKGBVbmtub3duIHJvdW5kaW5nTW9kZSAke3JvdW5kaW5nTW9kZX1gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gdHVwbGVWYWx1ZXNBcmVPbmUocGFyYW06IG51bWJlcnxudW1iZXJbXSk6IGJvb2xlYW4ge1xuICBjb25zdCBbZGltQSwgZGltQiwgZGltQ10gPSBwYXJzZVR1cGxlUGFyYW0ocGFyYW0pO1xuICByZXR1cm4gZGltQSA9PT0gMSAmJiBkaW1CID09PSAxICYmIGRpbUMgPT09IDE7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBlaXRoZXJTdHJpZGVzT3JEaWxhdGlvbnNBcmVPbmUoXG4gICAgc3RyaWRlczogbnVtYmVyfG51bWJlcltdLCBkaWxhdGlvbnM6IG51bWJlcnxudW1iZXJbXSk6IGJvb2xlYW4ge1xuICByZXR1cm4gdHVwbGVWYWx1ZXNBcmVPbmUoc3RyaWRlcykgfHwgdHVwbGVWYWx1ZXNBcmVPbmUoZGlsYXRpb25zKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHN0cmlkZXNPckRpbGF0aW9uc0FyZVBvc2l0aXZlKHZhbHVlczogbnVtYmVyfFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG51bWJlcltdKTogYm9vbGVhbiB7XG4gIHJldHVybiBwYXJzZVR1cGxlUGFyYW0odmFsdWVzKS5ldmVyeSh2YWx1ZSA9PiB2YWx1ZSA+IDApO1xufVxuXG4vKipcbiAqIENvbnZlcnQgQ29udjJEIGRhdGFGb3JtYXQgZnJvbSAnTkhXQyd8J05DSFcnIHRvXG4gKiAgICAnY2hhbm5lbHNMYXN0J3wnY2hhbm5lbHNGaXJzdCdcbiAqIEBwYXJhbSBkYXRhRm9ybWF0IGluICdOSFdDJ3wnTkNIVycgbW9kZVxuICogQHJldHVybiBkYXRhRm9ybWF0IGluICdjaGFubmVsc0xhc3QnfCdjaGFubmVsc0ZpcnN0JyBtb2RlXG4gKiBAdGhyb3dzIHVua25vd24gZGF0YUZvcm1hdFxuICovXG5leHBvcnQgZnVuY3Rpb24gY29udmVydENvbnYyRERhdGFGb3JtYXQoZGF0YUZvcm1hdDogJ05IV0MnfCdOQ0hXJyk6XG4gICAgJ2NoYW5uZWxzTGFzdCd8J2NoYW5uZWxzRmlyc3QnIHtcbiAgaWYgKGRhdGFGb3JtYXQgPT09ICdOSFdDJykge1xuICAgIHJldHVybiAnY2hhbm5lbHNMYXN0JztcbiAgfSBlbHNlIGlmIChkYXRhRm9ybWF0ID09PSAnTkNIVycpIHtcbiAgICByZXR1cm4gJ2NoYW5uZWxzRmlyc3QnO1xuICB9IGVsc2Uge1xuICAgIHRocm93IG5ldyBFcnJvcihgVW5rbm93biBkYXRhRm9ybWF0ICR7ZGF0YUZvcm1hdH1gKTtcbiAgfVxufVxuXG4vKipcbiAqIENoZWNrIHZhbGlkaXR5IG9mIHBhZCB3aGVuIHVzaW5nIGRpbVJvdW5kaW5nTW9kZS5cbiAqIEBwYXJhbSBvcERlc2MgQSBzdHJpbmcgb2Ygb3AgZGVzY3JpcHRpb25cbiAqIEBwYXJhbSBwYWQgVGhlIHR5cGUgb2YgcGFkZGluZyBhbGdvcml0aG0uXG4gKiAgIC0gYHNhbWVgIGFuZCBzdHJpZGUgMTogb3V0cHV0IHdpbGwgYmUgb2Ygc2FtZSBzaXplIGFzIGlucHV0LFxuICogICAgICAgcmVnYXJkbGVzcyBvZiBmaWx0ZXIgc2l6ZS5cbiAqICAgLSBgdmFsaWRgIG91dHB1dCB3aWxsIGJlIHNtYWxsZXIgdGhhbiBpbnB1dCBpZiBmaWx0ZXIgaXMgbGFyZ2VyXG4gKiAgICAgICB0aGFuIDF4MS5cbiAqICAgLSBGb3IgbW9yZSBpbmZvLCBzZWUgdGhpcyBndWlkZTpcbiAqICAgICBbaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2NvbnZvbHV0aW9uXShcbiAqICAgICAgICAgIGh0dHBzOi8vd3d3LnRlbnNvcmZsb3cub3JnL2FwaV9kb2NzL3B5dGhvbi90Zi9ubi9jb252b2x1dGlvbilcbiAqIEBwYXJhbSBkaW1Sb3VuZGluZ01vZGUgQSBzdHJpbmcgZnJvbTogJ2NlaWwnLCAncm91bmQnLCAnZmxvb3InLiBJZiBub25lIGlzXG4gKiAgICAgcHJvdmlkZWQsIGl0IHdpbGwgZGVmYXVsdCB0byB0cnVuY2F0ZS5cbiAqIEB0aHJvd3MgdW5rbm93biBwYWRkaW5nIHBhcmFtZXRlclxuICovXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tQYWRPbkRpbVJvdW5kaW5nTW9kZShcbiAgICBvcERlc2M6IHN0cmluZywgcGFkOiAndmFsaWQnfCdzYW1lJ3xudW1iZXJ8RXhwbGljaXRQYWRkaW5nLFxuICAgIGRpbVJvdW5kaW5nTW9kZT86ICdmbG9vcid8J3JvdW5kJ3wnY2VpbCcpIHtcbiAgaWYgKGRpbVJvdW5kaW5nTW9kZSAhPSBudWxsKSB7XG4gICAgaWYgKHR5cGVvZiBwYWQgPT09ICdzdHJpbmcnKSB7XG4gICAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgICBgRXJyb3IgaW4gJHtvcERlc2N9OiBwYWQgbXVzdCBiZSBhbiBpbnRlZ2VyIHdoZW4gdXNpbmcgYCArXG4gICAgICAgICAgYGRpbVJvdW5kaW5nTW9kZSAke2RpbVJvdW5kaW5nTW9kZX0gYnV0IGdvdCBwYWQgJHtwYWR9LmApO1xuICAgIH0gZWxzZSBpZiAodHlwZW9mIHBhZCA9PT0gJ251bWJlcicpIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIHV0aWwuaXNJbnQocGFkKSxcbiAgICAgICAgICAoKSA9PiBgRXJyb3IgaW4gJHtvcERlc2N9OiBwYWQgbXVzdCBiZSBhbiBpbnRlZ2VyIHdoZW4gdXNpbmcgYCArXG4gICAgICAgICAgICAgIGBkaW1Sb3VuZGluZ01vZGUgJHtkaW1Sb3VuZGluZ01vZGV9IGJ1dCBnb3QgcGFkICR7cGFkfS5gKTtcbiAgICB9IGVsc2UgaWYgKHR5cGVvZiBwYWQgPT09ICdvYmplY3QnKSB7XG4gICAgICAocGFkIGFzIEV4cGxpY2l0UGFkZGluZykuZm9yRWFjaChwID0+IHtcbiAgICAgICAgcC5mb3JFYWNoKHYgPT4ge1xuICAgICAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgICAgICB1dGlsLmlzSW50KHYpLFxuICAgICAgICAgICAgICAoKSA9PiBgRXJyb3IgaW4gJHtvcERlc2N9OiBwYWQgbXVzdCBiZSBhbiBpbnRlZ2VyIHdoZW4gdXNpbmcgYCArXG4gICAgICAgICAgICAgICAgICBgZGltUm91bmRpbmdNb2RlICR7ZGltUm91bmRpbmdNb2RlfSBidXQgZ290IHBhZCAke3Z9LmApO1xuICAgICAgICB9KTtcbiAgICAgIH0pO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aHJvdyBFcnJvcihgRXJyb3IgaW4gJHtvcERlc2N9OiBVbmtub3duIHBhZGRpbmcgcGFyYW1ldGVyOiAke3BhZH1gKTtcbiAgICB9XG4gIH1cbn1cbiJdfQ==