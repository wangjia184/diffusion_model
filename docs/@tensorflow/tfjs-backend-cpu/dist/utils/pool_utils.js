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
import { buffer } from '@tensorflow/tfjs-core';
export function pool(xValues, xShape, dtype, strides, convInfo, poolType) {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
        Number.POSITIVE_INFINITY);
    const output = buffer(convInfo.outShape, dtype);
    const outputVals = output.values;
    const outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] * convInfo.outShape[3];
    const outputRowStrides = convInfo.outShape[2] * convInfo.outShape[3];
    const outputColStrides = convInfo.outShape[3];
    for (let b = 0; b < convInfo.batchSize; ++b) {
        const outputBatchOffset = b * outputBatchStrides;
        const inputBatchOffset = b * strides[0];
        for (let d = 0; d < convInfo.inChannels; ++d) {
            for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                const xRCorner = yR * strideHeight - padTop;
                const xRMin = Math.max(0, xRCorner);
                const xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                const outputRowOffset = outputBatchOffset + yR * outputRowStrides;
                for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                    const xCCorner = yC * strideWidth - padLeft;
                    const xCMin = Math.max(0, xCCorner);
                    const xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                    let minMaxValue = initialValue;
                    let avgValue = 0;
                    let count = 0;
                    for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
                        const xROffset = inputBatchOffset + xR * strides[1];
                        for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
                            const xCOffset = xROffset + xC * strides[2];
                            const pixel = xValues[xCOffset + d];
                            if ((poolType === 'max' && pixel > minMaxValue)) {
                                minMaxValue = pixel;
                            }
                            else if (poolType === 'avg') {
                                avgValue += pixel;
                                count++;
                            }
                        }
                        if (isNaN(minMaxValue)) {
                            break;
                        }
                    }
                    const outputOffset = outputRowOffset + yC * outputColStrides + d;
                    outputVals[outputOffset] =
                        poolType === 'avg' ? avgValue / count : minMaxValue;
                }
            }
        }
    }
    return output;
}
export function maxPoolPositions(xValues, xShape, dtype, convInfo, flattenPositions = false, includeBatchInIndex = false) {
    const maxPositions = buffer(convInfo.outShape, 'int32');
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const xBuf = buffer(xShape, dtype, xValues);
    for (let b = 0; b < convInfo.batchSize; ++b) {
        for (let d = 0; d < convInfo.inChannels; ++d) {
            for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                const xRCorner = yR * strideHeight - padTop;
                let xRMin = xRCorner;
                while (xRMin < 0) {
                    xRMin += dilationHeight;
                }
                // const xRMin = Math.max(0, xRCorner);
                const xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                    const xCCorner = yC * strideWidth - padLeft;
                    let xCMin = xCCorner;
                    while (xCMin < 0) {
                        xCMin += dilationWidth;
                    }
                    const xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                    let maxValue = Number.NEGATIVE_INFINITY;
                    let maxPosition = -1;
                    for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
                        const wR = xR - xRCorner;
                        for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
                            const wC = xC - xCCorner;
                            const pixel = xBuf.get(b, xR, xC, d);
                            if (pixel > maxValue) {
                                maxValue = pixel;
                                if (flattenPositions) {
                                    maxPosition = includeBatchInIndex ?
                                        ((b * convInfo.inHeight + xR) * convInfo.inWidth + xC) *
                                            convInfo.inChannels +
                                            d :
                                        (xR * convInfo.inWidth + xC) * convInfo.inChannels + d;
                                }
                                else {
                                    maxPosition = wR * effectiveFilterWidth + wC;
                                }
                            }
                        }
                    }
                    maxPositions.set(maxPosition, b, yR, yC, d);
                }
            }
        }
    }
    return maxPositions;
}
export function pool3d(xValues, xShape, dtype, strides, convInfo, poolType) {
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
        Number.POSITIVE_INFINITY);
    const output = buffer(convInfo.outShape, dtype);
    const outputVals = output.values;
    const outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] *
        convInfo.outShape[3] * convInfo.outShape[4];
    const outputDepthStrides = convInfo.outShape[2] * convInfo.outShape[3] * convInfo.outShape[4];
    const outputRowStrides = convInfo.outShape[3] * convInfo.outShape[4];
    const outputColStrides = convInfo.outShape[4];
    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
        const outputBatchOffset = batch * outputBatchStrides;
        const inputBatchOffset = batch * strides[0];
        for (let channel = 0; channel < convInfo.inChannels; ++channel) {
            for (let yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
                const xDepthCorner = yDepth * strideDepth - padFront;
                let xDepthMin = xDepthCorner;
                while (xDepthMin < 0) {
                    xDepthMin += dilationDepth;
                }
                const xDepthMax = Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
                const outputDepthOffset = outputBatchOffset + yDepth * outputDepthStrides;
                for (let yRow = 0; yRow < convInfo.outHeight; ++yRow) {
                    const xRowCorner = yRow * strideHeight - padTop;
                    let xRowMin = xRowCorner;
                    while (xRowMin < 0) {
                        xRowMin += dilationHeight;
                    }
                    const xRowMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
                    const outputRowOffset = outputDepthOffset + yRow * outputRowStrides;
                    for (let yCol = 0; yCol < convInfo.outWidth; ++yCol) {
                        const xColCorner = yCol * strideWidth - padLeft;
                        let xColMin = xColCorner;
                        while (xColMin < 0) {
                            xColMin += dilationWidth;
                        }
                        const xColMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
                        // Shader code begins
                        const outputColOffset = outputRowOffset + yCol * outputColStrides;
                        let minMaxValue = initialValue;
                        let avgValue = 0;
                        let count = 0;
                        for (let xDepth = xDepthMin; xDepth < xDepthMax; xDepth += dilationDepth) {
                            const xDepthOffset = inputBatchOffset + xDepth * strides[1];
                            for (let xRow = xRowMin; xRow < xRowMax; xRow += dilationHeight) {
                                const xRowOffset = xDepthOffset + xRow * strides[2];
                                for (let xCol = xColMin; xCol < xColMax; xCol += dilationWidth) {
                                    const xColOffset = xRowOffset + xCol * strides[3];
                                    const pixel = xValues[xColOffset + channel];
                                    if ((poolType === 'max' && pixel > minMaxValue)) {
                                        minMaxValue = pixel;
                                    }
                                    else if (poolType === 'avg') {
                                        avgValue += pixel;
                                        count++;
                                    }
                                    if (isNaN(minMaxValue)) {
                                        break;
                                    }
                                }
                                if (isNaN(minMaxValue)) {
                                    break;
                                }
                            }
                            if (isNaN(minMaxValue)) {
                                break;
                            }
                        }
                        const outputOffset = outputColOffset + channel;
                        outputVals[outputOffset] = poolType === 'avg' ?
                            avgValue / Math.max(count, 1) :
                            minMaxValue;
                    }
                }
            }
        }
    }
    return output;
}
export function maxPool3dPositions(xBuf, convInfo) {
    const maxPositions = buffer(convInfo.outShape, 'int32');
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
        for (let channel = 0; channel < convInfo.inChannels; ++channel) {
            for (let yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
                const xDepthCorner = yDepth * strideDepth - padFront;
                let xDepthMin = xDepthCorner;
                while (xDepthMin < 0) {
                    xDepthMin += dilationDepth;
                }
                const xDepthMax = Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
                for (let yRow = 0; yRow < convInfo.outHeight; ++yRow) {
                    const xRowCorner = yRow * strideHeight - padTop;
                    let xRowMin = xRowCorner;
                    while (xRowMin < 0) {
                        xRowMin += dilationHeight;
                    }
                    const xRowMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
                    for (let yCol = 0; yCol < convInfo.outWidth; ++yCol) {
                        const xColCorner = yCol * strideWidth - padLeft;
                        let xColMin = xColCorner;
                        while (xColMin < 0) {
                            xColMin += dilationWidth;
                        }
                        const xColMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
                        // Shader code begins
                        let maxValue = Number.NEGATIVE_INFINITY;
                        let maxPosition = -1;
                        for (let xDepth = xDepthMin; xDepth < xDepthMax; xDepth += dilationDepth) {
                            const wDepth = xDepth - xDepthCorner;
                            for (let xRow = xRowMin; xRow < xRowMax; xRow += dilationHeight) {
                                const wRow = xRow - xRowCorner;
                                for (let xCol = xColMin; xCol < xColMax; xCol += dilationWidth) {
                                    const wCol = xCol - xColCorner;
                                    const pixel = xBuf.get(batch, xDepth, xRow, xCol, channel);
                                    if (pixel >= maxValue) {
                                        maxValue = pixel;
                                        maxPosition =
                                            wDepth * effectiveFilterHeight * effectiveFilterWidth +
                                                wRow * effectiveFilterHeight + wCol;
                                    }
                                }
                            }
                        }
                        maxPositions.set(maxPosition, batch, yDepth, yRow, yCol, channel);
                    }
                }
            }
        }
    }
    return maxPositions;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9vbF91dGlscy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL3V0aWxzL3Bvb2xfdXRpbHMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFlLE1BQU0sRUFBMkMsTUFBTSx1QkFBdUIsQ0FBQztBQUVyRyxNQUFNLFVBQVUsSUFBSSxDQUNoQixPQUFtQixFQUFFLE1BQWdCLEVBQUUsS0FBZSxFQUFFLE9BQWlCLEVBQ3pFLFFBQWlDLEVBQ2pDLFFBQXFCO0lBQ3ZCLE1BQU0sWUFBWSxHQUFHLFFBQVEsQ0FBQyxZQUFZLENBQUM7SUFDM0MsTUFBTSxXQUFXLEdBQUcsUUFBUSxDQUFDLFdBQVcsQ0FBQztJQUN6QyxNQUFNLGNBQWMsR0FBRyxRQUFRLENBQUMsY0FBYyxDQUFDO0lBQy9DLE1BQU0sYUFBYSxHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUM7SUFDN0MsTUFBTSxxQkFBcUIsR0FBRyxRQUFRLENBQUMscUJBQXFCLENBQUM7SUFDN0QsTUFBTSxvQkFBb0IsR0FBRyxRQUFRLENBQUMsb0JBQW9CLENBQUM7SUFDM0QsTUFBTSxNQUFNLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUM7SUFDcEMsTUFBTSxPQUFPLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUM7SUFFdEMsTUFBTSxZQUFZLEdBQ2QsQ0FBQyxRQUFRLEtBQUssS0FBSyxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsaUJBQWlCLENBQUMsQ0FBQztRQUMxQixNQUFNLENBQUMsaUJBQWlCLENBQUMsQ0FBQztJQUVwRCxNQUFNLE1BQU0sR0FBRyxNQUFNLENBQUMsUUFBUSxDQUFDLFFBQVEsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUNoRCxNQUFNLFVBQVUsR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDO0lBRWpDLE1BQU0sa0JBQWtCLEdBQ3BCLFFBQVEsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3ZFLE1BQU0sZ0JBQWdCLEdBQUcsUUFBUSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3JFLE1BQU0sZ0JBQWdCLEdBQUcsUUFBUSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUU5QyxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsUUFBUSxDQUFDLFNBQVMsRUFBRSxFQUFFLENBQUMsRUFBRTtRQUMzQyxNQUFNLGlCQUFpQixHQUFHLENBQUMsR0FBRyxrQkFBa0IsQ0FBQztRQUNqRCxNQUFNLGdCQUFnQixHQUFHLENBQUMsR0FBRyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDeEMsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxVQUFVLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDNUMsS0FBSyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLFFBQVEsQ0FBQyxTQUFTLEVBQUUsRUFBRSxFQUFFLEVBQUU7Z0JBQzlDLE1BQU0sUUFBUSxHQUFHLEVBQUUsR0FBRyxZQUFZLEdBQUcsTUFBTSxDQUFDO2dCQUM1QyxNQUFNLEtBQUssR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxRQUFRLENBQUMsQ0FBQztnQkFDcEMsTUFBTSxLQUFLLEdBQ1AsSUFBSSxDQUFDLEdBQUcsQ0FBQyxRQUFRLENBQUMsUUFBUSxFQUFFLHFCQUFxQixHQUFHLFFBQVEsQ0FBQyxDQUFDO2dCQUNsRSxNQUFNLGVBQWUsR0FBRyxpQkFBaUIsR0FBRyxFQUFFLEdBQUcsZ0JBQWdCLENBQUM7Z0JBQ2xFLEtBQUssSUFBSSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEVBQUUsR0FBRyxRQUFRLENBQUMsUUFBUSxFQUFFLEVBQUUsRUFBRSxFQUFFO29CQUM3QyxNQUFNLFFBQVEsR0FBRyxFQUFFLEdBQUcsV0FBVyxHQUFHLE9BQU8sQ0FBQztvQkFDNUMsTUFBTSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsUUFBUSxDQUFDLENBQUM7b0JBQ3BDLE1BQU0sS0FBSyxHQUNQLElBQUksQ0FBQyxHQUFHLENBQUMsUUFBUSxDQUFDLE9BQU8sRUFBRSxvQkFBb0IsR0FBRyxRQUFRLENBQUMsQ0FBQztvQkFDaEUsSUFBSSxXQUFXLEdBQUcsWUFBWSxDQUFDO29CQUMvQixJQUFJLFFBQVEsR0FBRyxDQUFDLENBQUM7b0JBQ2pCLElBQUksS0FBSyxHQUFHLENBQUMsQ0FBQztvQkFDZCxLQUFLLElBQUksRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsSUFBSSxjQUFjLEVBQUU7d0JBQ3JELE1BQU0sUUFBUSxHQUFHLGdCQUFnQixHQUFHLEVBQUUsR0FBRyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7d0JBQ3BELEtBQUssSUFBSSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsR0FBRyxLQUFLLEVBQUUsRUFBRSxJQUFJLGFBQWEsRUFBRTs0QkFDcEQsTUFBTSxRQUFRLEdBQUcsUUFBUSxHQUFHLEVBQUUsR0FBRyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7NEJBQzVDLE1BQU0sS0FBSyxHQUFHLE9BQU8sQ0FBQyxRQUFRLEdBQUcsQ0FBQyxDQUFDLENBQUM7NEJBQ3BDLElBQUksQ0FBQyxRQUFRLEtBQUssS0FBSyxJQUFJLEtBQUssR0FBRyxXQUFXLENBQUMsRUFBRTtnQ0FDL0MsV0FBVyxHQUFHLEtBQUssQ0FBQzs2QkFDckI7aUNBQU0sSUFBSSxRQUFRLEtBQUssS0FBSyxFQUFFO2dDQUM3QixRQUFRLElBQUksS0FBSyxDQUFDO2dDQUNsQixLQUFLLEVBQUUsQ0FBQzs2QkFDVDt5QkFDRjt3QkFDRCxJQUFJLEtBQUssQ0FBQyxXQUFXLENBQUMsRUFBRTs0QkFDdEIsTUFBTTt5QkFDUDtxQkFDRjtvQkFDRCxNQUFNLFlBQVksR0FBRyxlQUFlLEdBQUcsRUFBRSxHQUFHLGdCQUFnQixHQUFHLENBQUMsQ0FBQztvQkFDakUsVUFBVSxDQUFDLFlBQVksQ0FBQzt3QkFDcEIsUUFBUSxLQUFLLEtBQUssQ0FBQyxDQUFDLENBQUMsUUFBUSxHQUFHLEtBQUssQ0FBQyxDQUFDLENBQUMsV0FBVyxDQUFDO2lCQUN6RDthQUNGO1NBQ0Y7S0FDRjtJQUNELE9BQU8sTUFBTSxDQUFDO0FBQ2hCLENBQUM7QUFFRCxNQUFNLFVBQVUsZ0JBQWdCLENBQzVCLE9BQW1CLEVBQUUsTUFBZ0IsRUFBRSxLQUFlLEVBQ3RELFFBQWlDLEVBQUUsZ0JBQWdCLEdBQUcsS0FBSyxFQUMzRCxtQkFBbUIsR0FBRyxLQUFLO0lBQzdCLE1BQU0sWUFBWSxHQUFHLE1BQU0sQ0FBQyxRQUFRLENBQUMsUUFBUSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ3hELE1BQU0sWUFBWSxHQUFHLFFBQVEsQ0FBQyxZQUFZLENBQUM7SUFDM0MsTUFBTSxXQUFXLEdBQUcsUUFBUSxDQUFDLFdBQVcsQ0FBQztJQUN6QyxNQUFNLGNBQWMsR0FBRyxRQUFRLENBQUMsY0FBYyxDQUFDO0lBQy9DLE1BQU0sYUFBYSxHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUM7SUFDN0MsTUFBTSxxQkFBcUIsR0FBRyxRQUFRLENBQUMscUJBQXFCLENBQUM7SUFDN0QsTUFBTSxvQkFBb0IsR0FBRyxRQUFRLENBQUMsb0JBQW9CLENBQUM7SUFDM0QsTUFBTSxNQUFNLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUM7SUFDcEMsTUFBTSxPQUFPLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUM7SUFFdEMsTUFBTSxJQUFJLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxLQUFLLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFDNUMsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxTQUFTLEVBQUUsRUFBRSxDQUFDLEVBQUU7UUFDM0MsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxVQUFVLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDNUMsS0FBSyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLFFBQVEsQ0FBQyxTQUFTLEVBQUUsRUFBRSxFQUFFLEVBQUU7Z0JBQzlDLE1BQU0sUUFBUSxHQUFHLEVBQUUsR0FBRyxZQUFZLEdBQUcsTUFBTSxDQUFDO2dCQUM1QyxJQUFJLEtBQUssR0FBRyxRQUFRLENBQUM7Z0JBQ3JCLE9BQU8sS0FBSyxHQUFHLENBQUMsRUFBRTtvQkFDaEIsS0FBSyxJQUFJLGNBQWMsQ0FBQztpQkFDekI7Z0JBQ0QsdUNBQXVDO2dCQUN2QyxNQUFNLEtBQUssR0FDUCxJQUFJLENBQUMsR0FBRyxDQUFDLFFBQVEsQ0FBQyxRQUFRLEVBQUUscUJBQXFCLEdBQUcsUUFBUSxDQUFDLENBQUM7Z0JBQ2xFLEtBQUssSUFBSSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEVBQUUsR0FBRyxRQUFRLENBQUMsUUFBUSxFQUFFLEVBQUUsRUFBRSxFQUFFO29CQUM3QyxNQUFNLFFBQVEsR0FBRyxFQUFFLEdBQUcsV0FBVyxHQUFHLE9BQU8sQ0FBQztvQkFDNUMsSUFBSSxLQUFLLEdBQUcsUUFBUSxDQUFDO29CQUNyQixPQUFPLEtBQUssR0FBRyxDQUFDLEVBQUU7d0JBQ2hCLEtBQUssSUFBSSxhQUFhLENBQUM7cUJBQ3hCO29CQUNELE1BQU0sS0FBSyxHQUNQLElBQUksQ0FBQyxHQUFHLENBQUMsUUFBUSxDQUFDLE9BQU8sRUFBRSxvQkFBb0IsR0FBRyxRQUFRLENBQUMsQ0FBQztvQkFDaEUsSUFBSSxRQUFRLEdBQUcsTUFBTSxDQUFDLGlCQUFpQixDQUFDO29CQUN4QyxJQUFJLFdBQVcsR0FBRyxDQUFDLENBQUMsQ0FBQztvQkFFckIsS0FBSyxJQUFJLEVBQUUsR0FBRyxLQUFLLEVBQUUsRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLElBQUksY0FBYyxFQUFFO3dCQUNyRCxNQUFNLEVBQUUsR0FBRyxFQUFFLEdBQUcsUUFBUSxDQUFDO3dCQUN6QixLQUFLLElBQUksRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsSUFBSSxhQUFhLEVBQUU7NEJBQ3BELE1BQU0sRUFBRSxHQUFHLEVBQUUsR0FBRyxRQUFRLENBQUM7NEJBQ3pCLE1BQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUM7NEJBQ3JDLElBQUksS0FBSyxHQUFHLFFBQVEsRUFBRTtnQ0FDcEIsUUFBUSxHQUFHLEtBQWUsQ0FBQztnQ0FDM0IsSUFBSSxnQkFBZ0IsRUFBRTtvQ0FDcEIsV0FBVyxHQUFHLG1CQUFtQixDQUFDLENBQUM7d0NBQy9CLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLFFBQVEsR0FBRyxFQUFFLENBQUMsR0FBRyxRQUFRLENBQUMsT0FBTyxHQUFHLEVBQUUsQ0FBQzs0Q0FDOUMsUUFBUSxDQUFDLFVBQVU7NENBQ3ZCLENBQUMsQ0FBQyxDQUFDO3dDQUNQLENBQUMsRUFBRSxHQUFHLFFBQVEsQ0FBQyxPQUFPLEdBQUcsRUFBRSxDQUFDLEdBQUcsUUFBUSxDQUFDLFVBQVUsR0FBRyxDQUFDLENBQUM7aUNBQzVEO3FDQUFNO29DQUNMLFdBQVcsR0FBRyxFQUFFLEdBQUcsb0JBQW9CLEdBQUcsRUFBRSxDQUFDO2lDQUM5Qzs2QkFDRjt5QkFDRjtxQkFDRjtvQkFDRCxZQUFZLENBQUMsR0FBRyxDQUFDLFdBQVcsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQztpQkFDN0M7YUFDRjtTQUNGO0tBQ0Y7SUFDRCxPQUFPLFlBQVksQ0FBQztBQUN0QixDQUFDO0FBRUQsTUFBTSxVQUFVLE1BQU0sQ0FDbEIsT0FBbUIsRUFBRSxNQUFnQixFQUFFLEtBQWUsRUFBRSxPQUFpQixFQUN6RSxRQUFpQyxFQUNqQyxRQUFxQjtJQUN2QixNQUFNLFdBQVcsR0FBRyxRQUFRLENBQUMsV0FBVyxDQUFDO0lBQ3pDLE1BQU0sWUFBWSxHQUFHLFFBQVEsQ0FBQyxZQUFZLENBQUM7SUFDM0MsTUFBTSxXQUFXLEdBQUcsUUFBUSxDQUFDLFdBQVcsQ0FBQztJQUN6QyxNQUFNLGFBQWEsR0FBRyxRQUFRLENBQUMsYUFBYSxDQUFDO0lBQzdDLE1BQU0sY0FBYyxHQUFHLFFBQVEsQ0FBQyxjQUFjLENBQUM7SUFDL0MsTUFBTSxhQUFhLEdBQUcsUUFBUSxDQUFDLGFBQWEsQ0FBQztJQUM3QyxNQUFNLG9CQUFvQixHQUFHLFFBQVEsQ0FBQyxvQkFBb0IsQ0FBQztJQUMzRCxNQUFNLHFCQUFxQixHQUFHLFFBQVEsQ0FBQyxxQkFBcUIsQ0FBQztJQUM3RCxNQUFNLG9CQUFvQixHQUFHLFFBQVEsQ0FBQyxvQkFBb0IsQ0FBQztJQUMzRCxNQUFNLFFBQVEsR0FBRyxRQUFRLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQztJQUN4QyxNQUFNLE1BQU0sR0FBRyxRQUFRLENBQUMsT0FBTyxDQUFDLEdBQUcsQ0FBQztJQUNwQyxNQUFNLE9BQU8sR0FBRyxRQUFRLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQztJQUV0QyxNQUFNLFlBQVksR0FDZCxDQUFDLFFBQVEsS0FBSyxLQUFLLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDO1FBQzFCLE1BQU0sQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDO0lBRXBELE1BQU0sTUFBTSxHQUFHLE1BQU0sQ0FBQyxRQUFRLENBQUMsUUFBUSxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBQ2hELE1BQU0sVUFBVSxHQUFHLE1BQU0sQ0FBQyxNQUFNLENBQUM7SUFFakMsTUFBTSxrQkFBa0IsR0FBRyxRQUFRLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO1FBQ2xFLFFBQVEsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNoRCxNQUFNLGtCQUFrQixHQUNwQixRQUFRLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN2RSxNQUFNLGdCQUFnQixHQUFHLFFBQVEsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNyRSxNQUFNLGdCQUFnQixHQUFHLFFBQVEsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFFOUMsS0FBSyxJQUFJLEtBQUssR0FBRyxDQUFDLEVBQUUsS0FBSyxHQUFHLFFBQVEsQ0FBQyxTQUFTLEVBQUUsRUFBRSxLQUFLLEVBQUU7UUFDdkQsTUFBTSxpQkFBaUIsR0FBRyxLQUFLLEdBQUcsa0JBQWtCLENBQUM7UUFDckQsTUFBTSxnQkFBZ0IsR0FBRyxLQUFLLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzVDLEtBQUssSUFBSSxPQUFPLEdBQUcsQ0FBQyxFQUFFLE9BQU8sR0FBRyxRQUFRLENBQUMsVUFBVSxFQUFFLEVBQUUsT0FBTyxFQUFFO1lBQzlELEtBQUssSUFBSSxNQUFNLEdBQUcsQ0FBQyxFQUFFLE1BQU0sR0FBRyxRQUFRLENBQUMsUUFBUSxFQUFFLEVBQUUsTUFBTSxFQUFFO2dCQUN6RCxNQUFNLFlBQVksR0FBRyxNQUFNLEdBQUcsV0FBVyxHQUFHLFFBQVEsQ0FBQztnQkFDckQsSUFBSSxTQUFTLEdBQUcsWUFBWSxDQUFDO2dCQUM3QixPQUFPLFNBQVMsR0FBRyxDQUFDLEVBQUU7b0JBQ3BCLFNBQVMsSUFBSSxhQUFhLENBQUM7aUJBQzVCO2dCQUNELE1BQU0sU0FBUyxHQUNYLElBQUksQ0FBQyxHQUFHLENBQUMsUUFBUSxDQUFDLE9BQU8sRUFBRSxvQkFBb0IsR0FBRyxZQUFZLENBQUMsQ0FBQztnQkFDcEUsTUFBTSxpQkFBaUIsR0FDbkIsaUJBQWlCLEdBQUcsTUFBTSxHQUFHLGtCQUFrQixDQUFDO2dCQUNwRCxLQUFLLElBQUksSUFBSSxHQUFHLENBQUMsRUFBRSxJQUFJLEdBQUcsUUFBUSxDQUFDLFNBQVMsRUFBRSxFQUFFLElBQUksRUFBRTtvQkFDcEQsTUFBTSxVQUFVLEdBQUcsSUFBSSxHQUFHLFlBQVksR0FBRyxNQUFNLENBQUM7b0JBQ2hELElBQUksT0FBTyxHQUFHLFVBQVUsQ0FBQztvQkFDekIsT0FBTyxPQUFPLEdBQUcsQ0FBQyxFQUFFO3dCQUNsQixPQUFPLElBQUksY0FBYyxDQUFDO3FCQUMzQjtvQkFDRCxNQUFNLE9BQU8sR0FDVCxJQUFJLENBQUMsR0FBRyxDQUFDLFFBQVEsQ0FBQyxRQUFRLEVBQUUscUJBQXFCLEdBQUcsVUFBVSxDQUFDLENBQUM7b0JBQ3BFLE1BQU0sZUFBZSxHQUFHLGlCQUFpQixHQUFHLElBQUksR0FBRyxnQkFBZ0IsQ0FBQztvQkFDcEUsS0FBSyxJQUFJLElBQUksR0FBRyxDQUFDLEVBQUUsSUFBSSxHQUFHLFFBQVEsQ0FBQyxRQUFRLEVBQUUsRUFBRSxJQUFJLEVBQUU7d0JBQ25ELE1BQU0sVUFBVSxHQUFHLElBQUksR0FBRyxXQUFXLEdBQUcsT0FBTyxDQUFDO3dCQUNoRCxJQUFJLE9BQU8sR0FBRyxVQUFVLENBQUM7d0JBQ3pCLE9BQU8sT0FBTyxHQUFHLENBQUMsRUFBRTs0QkFDbEIsT0FBTyxJQUFJLGFBQWEsQ0FBQzt5QkFDMUI7d0JBQ0QsTUFBTSxPQUFPLEdBQ1QsSUFBSSxDQUFDLEdBQUcsQ0FBQyxRQUFRLENBQUMsT0FBTyxFQUFFLG9CQUFvQixHQUFHLFVBQVUsQ0FBQyxDQUFDO3dCQUNsRSxxQkFBcUI7d0JBQ3JCLE1BQU0sZUFBZSxHQUFHLGVBQWUsR0FBRyxJQUFJLEdBQUcsZ0JBQWdCLENBQUM7d0JBQ2xFLElBQUksV0FBVyxHQUFHLFlBQVksQ0FBQzt3QkFDL0IsSUFBSSxRQUFRLEdBQUcsQ0FBQyxDQUFDO3dCQUNqQixJQUFJLEtBQUssR0FBRyxDQUFDLENBQUM7d0JBQ2QsS0FBSyxJQUFJLE1BQU0sR0FBRyxTQUFTLEVBQUUsTUFBTSxHQUFHLFNBQVMsRUFDMUMsTUFBTSxJQUFJLGFBQWEsRUFBRTs0QkFDNUIsTUFBTSxZQUFZLEdBQUcsZ0JBQWdCLEdBQUcsTUFBTSxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQzs0QkFDNUQsS0FBSyxJQUFJLElBQUksR0FBRyxPQUFPLEVBQUUsSUFBSSxHQUFHLE9BQU8sRUFBRSxJQUFJLElBQUksY0FBYyxFQUFFO2dDQUMvRCxNQUFNLFVBQVUsR0FBRyxZQUFZLEdBQUcsSUFBSSxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQztnQ0FDcEQsS0FBSyxJQUFJLElBQUksR0FBRyxPQUFPLEVBQUUsSUFBSSxHQUFHLE9BQU8sRUFDbEMsSUFBSSxJQUFJLGFBQWEsRUFBRTtvQ0FDMUIsTUFBTSxVQUFVLEdBQUcsVUFBVSxHQUFHLElBQUksR0FBRyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7b0NBQ2xELE1BQU0sS0FBSyxHQUFHLE9BQU8sQ0FBQyxVQUFVLEdBQUcsT0FBTyxDQUFDLENBQUM7b0NBQzVDLElBQUksQ0FBQyxRQUFRLEtBQUssS0FBSyxJQUFJLEtBQUssR0FBRyxXQUFXLENBQUMsRUFBRTt3Q0FDL0MsV0FBVyxHQUFHLEtBQUssQ0FBQztxQ0FDckI7eUNBQU0sSUFBSSxRQUFRLEtBQUssS0FBSyxFQUFFO3dDQUM3QixRQUFRLElBQUksS0FBSyxDQUFDO3dDQUNsQixLQUFLLEVBQUUsQ0FBQztxQ0FDVDtvQ0FDRCxJQUFJLEtBQUssQ0FBQyxXQUFXLENBQUMsRUFBRTt3Q0FDdEIsTUFBTTtxQ0FDUDtpQ0FDRjtnQ0FDRCxJQUFJLEtBQUssQ0FBQyxXQUFXLENBQUMsRUFBRTtvQ0FDdEIsTUFBTTtpQ0FDUDs2QkFDRjs0QkFDRCxJQUFJLEtBQUssQ0FBQyxXQUFXLENBQUMsRUFBRTtnQ0FDdEIsTUFBTTs2QkFDUDt5QkFDRjt3QkFDRCxNQUFNLFlBQVksR0FBRyxlQUFlLEdBQUcsT0FBTyxDQUFDO3dCQUMvQyxVQUFVLENBQUMsWUFBWSxDQUFDLEdBQUcsUUFBUSxLQUFLLEtBQUssQ0FBQyxDQUFDOzRCQUMzQyxRQUFRLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQzs0QkFDL0IsV0FBVyxDQUFDO3FCQUNqQjtpQkFDRjthQUNGO1NBQ0Y7S0FDRjtJQUVELE9BQU8sTUFBTSxDQUFDO0FBQ2hCLENBQUM7QUFFRCxNQUFNLFVBQVUsa0JBQWtCLENBQzlCLElBQWtDLEVBQ2xDLFFBQWlDO0lBQ25DLE1BQU0sWUFBWSxHQUFHLE1BQU0sQ0FBQyxRQUFRLENBQUMsUUFBUSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ3hELE1BQU0sV0FBVyxHQUFHLFFBQVEsQ0FBQyxXQUFXLENBQUM7SUFDekMsTUFBTSxZQUFZLEdBQUcsUUFBUSxDQUFDLFlBQVksQ0FBQztJQUMzQyxNQUFNLFdBQVcsR0FBRyxRQUFRLENBQUMsV0FBVyxDQUFDO0lBQ3pDLE1BQU0sYUFBYSxHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUM7SUFDN0MsTUFBTSxjQUFjLEdBQUcsUUFBUSxDQUFDLGNBQWMsQ0FBQztJQUMvQyxNQUFNLGFBQWEsR0FBRyxRQUFRLENBQUMsYUFBYSxDQUFDO0lBQzdDLE1BQU0sb0JBQW9CLEdBQUcsUUFBUSxDQUFDLG9CQUFvQixDQUFDO0lBQzNELE1BQU0scUJBQXFCLEdBQUcsUUFBUSxDQUFDLHFCQUFxQixDQUFDO0lBQzdELE1BQU0sb0JBQW9CLEdBQUcsUUFBUSxDQUFDLG9CQUFvQixDQUFDO0lBQzNELE1BQU0sUUFBUSxHQUFHLFFBQVEsQ0FBQyxPQUFPLENBQUMsS0FBSyxDQUFDO0lBQ3hDLE1BQU0sTUFBTSxHQUFHLFFBQVEsQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDO0lBQ3BDLE1BQU0sT0FBTyxHQUFHLFFBQVEsQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDO0lBRXRDLEtBQUssSUFBSSxLQUFLLEdBQUcsQ0FBQyxFQUFFLEtBQUssR0FBRyxRQUFRLENBQUMsU0FBUyxFQUFFLEVBQUUsS0FBSyxFQUFFO1FBQ3ZELEtBQUssSUFBSSxPQUFPLEdBQUcsQ0FBQyxFQUFFLE9BQU8sR0FBRyxRQUFRLENBQUMsVUFBVSxFQUFFLEVBQUUsT0FBTyxFQUFFO1lBQzlELEtBQUssSUFBSSxNQUFNLEdBQUcsQ0FBQyxFQUFFLE1BQU0sR0FBRyxRQUFRLENBQUMsUUFBUSxFQUFFLEVBQUUsTUFBTSxFQUFFO2dCQUN6RCxNQUFNLFlBQVksR0FBRyxNQUFNLEdBQUcsV0FBVyxHQUFHLFFBQVEsQ0FBQztnQkFDckQsSUFBSSxTQUFTLEdBQUcsWUFBWSxDQUFDO2dCQUM3QixPQUFPLFNBQVMsR0FBRyxDQUFDLEVBQUU7b0JBQ3BCLFNBQVMsSUFBSSxhQUFhLENBQUM7aUJBQzVCO2dCQUNELE1BQU0sU0FBUyxHQUNYLElBQUksQ0FBQyxHQUFHLENBQUMsUUFBUSxDQUFDLE9BQU8sRUFBRSxvQkFBb0IsR0FBRyxZQUFZLENBQUMsQ0FBQztnQkFDcEUsS0FBSyxJQUFJLElBQUksR0FBRyxDQUFDLEVBQUUsSUFBSSxHQUFHLFFBQVEsQ0FBQyxTQUFTLEVBQUUsRUFBRSxJQUFJLEVBQUU7b0JBQ3BELE1BQU0sVUFBVSxHQUFHLElBQUksR0FBRyxZQUFZLEdBQUcsTUFBTSxDQUFDO29CQUNoRCxJQUFJLE9BQU8sR0FBRyxVQUFVLENBQUM7b0JBQ3pCLE9BQU8sT0FBTyxHQUFHLENBQUMsRUFBRTt3QkFDbEIsT0FBTyxJQUFJLGNBQWMsQ0FBQztxQkFDM0I7b0JBQ0QsTUFBTSxPQUFPLEdBQ1QsSUFBSSxDQUFDLEdBQUcsQ0FBQyxRQUFRLENBQUMsUUFBUSxFQUFFLHFCQUFxQixHQUFHLFVBQVUsQ0FBQyxDQUFDO29CQUNwRSxLQUFLLElBQUksSUFBSSxHQUFHLENBQUMsRUFBRSxJQUFJLEdBQUcsUUFBUSxDQUFDLFFBQVEsRUFBRSxFQUFFLElBQUksRUFBRTt3QkFDbkQsTUFBTSxVQUFVLEdBQUcsSUFBSSxHQUFHLFdBQVcsR0FBRyxPQUFPLENBQUM7d0JBQ2hELElBQUksT0FBTyxHQUFHLFVBQVUsQ0FBQzt3QkFDekIsT0FBTyxPQUFPLEdBQUcsQ0FBQyxFQUFFOzRCQUNsQixPQUFPLElBQUksYUFBYSxDQUFDO3lCQUMxQjt3QkFDRCxNQUFNLE9BQU8sR0FDVCxJQUFJLENBQUMsR0FBRyxDQUFDLFFBQVEsQ0FBQyxPQUFPLEVBQUUsb0JBQW9CLEdBQUcsVUFBVSxDQUFDLENBQUM7d0JBRWxFLHFCQUFxQjt3QkFDckIsSUFBSSxRQUFRLEdBQUcsTUFBTSxDQUFDLGlCQUFpQixDQUFDO3dCQUN4QyxJQUFJLFdBQVcsR0FBRyxDQUFDLENBQUMsQ0FBQzt3QkFFckIsS0FBSyxJQUFJLE1BQU0sR0FBRyxTQUFTLEVBQUUsTUFBTSxHQUFHLFNBQVMsRUFDMUMsTUFBTSxJQUFJLGFBQWEsRUFBRTs0QkFDNUIsTUFBTSxNQUFNLEdBQUcsTUFBTSxHQUFHLFlBQVksQ0FBQzs0QkFDckMsS0FBSyxJQUFJLElBQUksR0FBRyxPQUFPLEVBQUUsSUFBSSxHQUFHLE9BQU8sRUFBRSxJQUFJLElBQUksY0FBYyxFQUFFO2dDQUMvRCxNQUFNLElBQUksR0FBRyxJQUFJLEdBQUcsVUFBVSxDQUFDO2dDQUMvQixLQUFLLElBQUksSUFBSSxHQUFHLE9BQU8sRUFBRSxJQUFJLEdBQUcsT0FBTyxFQUNsQyxJQUFJLElBQUksYUFBYSxFQUFFO29DQUMxQixNQUFNLElBQUksR0FBRyxJQUFJLEdBQUcsVUFBVSxDQUFDO29DQUMvQixNQUFNLEtBQUssR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLEtBQUssRUFBRSxNQUFNLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxPQUFPLENBQUMsQ0FBQztvQ0FDM0QsSUFBSSxLQUFLLElBQUksUUFBUSxFQUFFO3dDQUNyQixRQUFRLEdBQUcsS0FBZSxDQUFDO3dDQUMzQixXQUFXOzRDQUNQLE1BQU0sR0FBRyxxQkFBcUIsR0FBRyxvQkFBb0I7Z0RBQ3JELElBQUksR0FBRyxxQkFBcUIsR0FBRyxJQUFJLENBQUM7cUNBQ3pDO2lDQUNGOzZCQUNGO3lCQUNGO3dCQUVELFlBQVksQ0FBQyxHQUFHLENBQUMsV0FBVyxFQUFFLEtBQUssRUFBRSxNQUFNLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxPQUFPLENBQUMsQ0FBQztxQkFDbkU7aUJBQ0Y7YUFDRjtTQUNGO0tBQ0Y7SUFFRCxPQUFPLFlBQVksQ0FBQztBQUN0QixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgYnVmZmVyLCBEYXRhVHlwZSwgUmFuaywgVGVuc29yQnVmZmVyLCBUeXBlZEFycmF5fSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5leHBvcnQgZnVuY3Rpb24gcG9vbChcbiAgICB4VmFsdWVzOiBUeXBlZEFycmF5LCB4U2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUsIHN0cmlkZXM6IG51bWJlcltdLFxuICAgIGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbyxcbiAgICBwb29sVHlwZTogJ21heCd8J2F2ZycpOiBUZW5zb3JCdWZmZXI8UmFuaywgRGF0YVR5cGU+IHtcbiAgY29uc3Qgc3RyaWRlSGVpZ2h0ID0gY29udkluZm8uc3RyaWRlSGVpZ2h0O1xuICBjb25zdCBzdHJpZGVXaWR0aCA9IGNvbnZJbmZvLnN0cmlkZVdpZHRoO1xuICBjb25zdCBkaWxhdGlvbkhlaWdodCA9IGNvbnZJbmZvLmRpbGF0aW9uSGVpZ2h0O1xuICBjb25zdCBkaWxhdGlvbldpZHRoID0gY29udkluZm8uZGlsYXRpb25XaWR0aDtcbiAgY29uc3QgZWZmZWN0aXZlRmlsdGVySGVpZ2h0ID0gY29udkluZm8uZWZmZWN0aXZlRmlsdGVySGVpZ2h0O1xuICBjb25zdCBlZmZlY3RpdmVGaWx0ZXJXaWR0aCA9IGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlcldpZHRoO1xuICBjb25zdCBwYWRUb3AgPSBjb252SW5mby5wYWRJbmZvLnRvcDtcbiAgY29uc3QgcGFkTGVmdCA9IGNvbnZJbmZvLnBhZEluZm8ubGVmdDtcblxuICBjb25zdCBpbml0aWFsVmFsdWUgPVxuICAgICAgKHBvb2xUeXBlID09PSAnbWF4JyA/IE51bWJlci5ORUdBVElWRV9JTkZJTklUWSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgTnVtYmVyLlBPU0lUSVZFX0lORklOSVRZKTtcblxuICBjb25zdCBvdXRwdXQgPSBidWZmZXIoY29udkluZm8ub3V0U2hhcGUsIGR0eXBlKTtcbiAgY29uc3Qgb3V0cHV0VmFscyA9IG91dHB1dC52YWx1ZXM7XG5cbiAgY29uc3Qgb3V0cHV0QmF0Y2hTdHJpZGVzID1cbiAgICAgIGNvbnZJbmZvLm91dFNoYXBlWzFdICogY29udkluZm8ub3V0U2hhcGVbMl0gKiBjb252SW5mby5vdXRTaGFwZVszXTtcbiAgY29uc3Qgb3V0cHV0Um93U3RyaWRlcyA9IGNvbnZJbmZvLm91dFNoYXBlWzJdICogY29udkluZm8ub3V0U2hhcGVbM107XG4gIGNvbnN0IG91dHB1dENvbFN0cmlkZXMgPSBjb252SW5mby5vdXRTaGFwZVszXTtcblxuICBmb3IgKGxldCBiID0gMDsgYiA8IGNvbnZJbmZvLmJhdGNoU2l6ZTsgKytiKSB7XG4gICAgY29uc3Qgb3V0cHV0QmF0Y2hPZmZzZXQgPSBiICogb3V0cHV0QmF0Y2hTdHJpZGVzO1xuICAgIGNvbnN0IGlucHV0QmF0Y2hPZmZzZXQgPSBiICogc3RyaWRlc1swXTtcbiAgICBmb3IgKGxldCBkID0gMDsgZCA8IGNvbnZJbmZvLmluQ2hhbm5lbHM7ICsrZCkge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IGNvbnZJbmZvLm91dEhlaWdodDsgKyt5Uikge1xuICAgICAgICBjb25zdCB4UkNvcm5lciA9IHlSICogc3RyaWRlSGVpZ2h0IC0gcGFkVG9wO1xuICAgICAgICBjb25zdCB4Uk1pbiA9IE1hdGgubWF4KDAsIHhSQ29ybmVyKTtcbiAgICAgICAgY29uc3QgeFJNYXggPVxuICAgICAgICAgICAgTWF0aC5taW4oY29udkluZm8uaW5IZWlnaHQsIGVmZmVjdGl2ZUZpbHRlckhlaWdodCArIHhSQ29ybmVyKTtcbiAgICAgICAgY29uc3Qgb3V0cHV0Um93T2Zmc2V0ID0gb3V0cHV0QmF0Y2hPZmZzZXQgKyB5UiAqIG91dHB1dFJvd1N0cmlkZXM7XG4gICAgICAgIGZvciAobGV0IHlDID0gMDsgeUMgPCBjb252SW5mby5vdXRXaWR0aDsgKyt5Qykge1xuICAgICAgICAgIGNvbnN0IHhDQ29ybmVyID0geUMgKiBzdHJpZGVXaWR0aCAtIHBhZExlZnQ7XG4gICAgICAgICAgY29uc3QgeENNaW4gPSBNYXRoLm1heCgwLCB4Q0Nvcm5lcik7XG4gICAgICAgICAgY29uc3QgeENNYXggPVxuICAgICAgICAgICAgICBNYXRoLm1pbihjb252SW5mby5pbldpZHRoLCBlZmZlY3RpdmVGaWx0ZXJXaWR0aCArIHhDQ29ybmVyKTtcbiAgICAgICAgICBsZXQgbWluTWF4VmFsdWUgPSBpbml0aWFsVmFsdWU7XG4gICAgICAgICAgbGV0IGF2Z1ZhbHVlID0gMDtcbiAgICAgICAgICBsZXQgY291bnQgPSAwO1xuICAgICAgICAgIGZvciAobGV0IHhSID0geFJNaW47IHhSIDwgeFJNYXg7IHhSICs9IGRpbGF0aW9uSGVpZ2h0KSB7XG4gICAgICAgICAgICBjb25zdCB4Uk9mZnNldCA9IGlucHV0QmF0Y2hPZmZzZXQgKyB4UiAqIHN0cmlkZXNbMV07XG4gICAgICAgICAgICBmb3IgKGxldCB4QyA9IHhDTWluOyB4QyA8IHhDTWF4OyB4QyArPSBkaWxhdGlvbldpZHRoKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHhDT2Zmc2V0ID0geFJPZmZzZXQgKyB4QyAqIHN0cmlkZXNbMl07XG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geFZhbHVlc1t4Q09mZnNldCArIGRdO1xuICAgICAgICAgICAgICBpZiAoKHBvb2xUeXBlID09PSAnbWF4JyAmJiBwaXhlbCA+IG1pbk1heFZhbHVlKSkge1xuICAgICAgICAgICAgICAgIG1pbk1heFZhbHVlID0gcGl4ZWw7XG4gICAgICAgICAgICAgIH0gZWxzZSBpZiAocG9vbFR5cGUgPT09ICdhdmcnKSB7XG4gICAgICAgICAgICAgICAgYXZnVmFsdWUgKz0gcGl4ZWw7XG4gICAgICAgICAgICAgICAgY291bnQrKztcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgaWYgKGlzTmFOKG1pbk1heFZhbHVlKSkge1xuICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgY29uc3Qgb3V0cHV0T2Zmc2V0ID0gb3V0cHV0Um93T2Zmc2V0ICsgeUMgKiBvdXRwdXRDb2xTdHJpZGVzICsgZDtcbiAgICAgICAgICBvdXRwdXRWYWxzW291dHB1dE9mZnNldF0gPVxuICAgICAgICAgICAgICBwb29sVHlwZSA9PT0gJ2F2ZycgPyBhdmdWYWx1ZSAvIGNvdW50IDogbWluTWF4VmFsdWU7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cbiAgcmV0dXJuIG91dHB1dDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIG1heFBvb2xQb3NpdGlvbnMoXG4gICAgeFZhbHVlczogVHlwZWRBcnJheSwgeFNoYXBlOiBudW1iZXJbXSwgZHR5cGU6IERhdGFUeXBlLFxuICAgIGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbywgZmxhdHRlblBvc2l0aW9ucyA9IGZhbHNlLFxuICAgIGluY2x1ZGVCYXRjaEluSW5kZXggPSBmYWxzZSk6IFRlbnNvckJ1ZmZlcjxSYW5rLCAnaW50MzInPiB7XG4gIGNvbnN0IG1heFBvc2l0aW9ucyA9IGJ1ZmZlcihjb252SW5mby5vdXRTaGFwZSwgJ2ludDMyJyk7XG4gIGNvbnN0IHN0cmlkZUhlaWdodCA9IGNvbnZJbmZvLnN0cmlkZUhlaWdodDtcbiAgY29uc3Qgc3RyaWRlV2lkdGggPSBjb252SW5mby5zdHJpZGVXaWR0aDtcbiAgY29uc3QgZGlsYXRpb25IZWlnaHQgPSBjb252SW5mby5kaWxhdGlvbkhlaWdodDtcbiAgY29uc3QgZGlsYXRpb25XaWR0aCA9IGNvbnZJbmZvLmRpbGF0aW9uV2lkdGg7XG4gIGNvbnN0IGVmZmVjdGl2ZUZpbHRlckhlaWdodCA9IGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlckhlaWdodDtcbiAgY29uc3QgZWZmZWN0aXZlRmlsdGVyV2lkdGggPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJXaWR0aDtcbiAgY29uc3QgcGFkVG9wID0gY29udkluZm8ucGFkSW5mby50b3A7XG4gIGNvbnN0IHBhZExlZnQgPSBjb252SW5mby5wYWRJbmZvLmxlZnQ7XG5cbiAgY29uc3QgeEJ1ZiA9IGJ1ZmZlcih4U2hhcGUsIGR0eXBlLCB4VmFsdWVzKTtcbiAgZm9yIChsZXQgYiA9IDA7IGIgPCBjb252SW5mby5iYXRjaFNpemU7ICsrYikge1xuICAgIGZvciAobGV0IGQgPSAwOyBkIDwgY29udkluZm8uaW5DaGFubmVsczsgKytkKSB7XG4gICAgICBmb3IgKGxldCB5UiA9IDA7IHlSIDwgY29udkluZm8ub3V0SGVpZ2h0OyArK3lSKSB7XG4gICAgICAgIGNvbnN0IHhSQ29ybmVyID0geVIgKiBzdHJpZGVIZWlnaHQgLSBwYWRUb3A7XG4gICAgICAgIGxldCB4Uk1pbiA9IHhSQ29ybmVyO1xuICAgICAgICB3aGlsZSAoeFJNaW4gPCAwKSB7XG4gICAgICAgICAgeFJNaW4gKz0gZGlsYXRpb25IZWlnaHQ7XG4gICAgICAgIH1cbiAgICAgICAgLy8gY29uc3QgeFJNaW4gPSBNYXRoLm1heCgwLCB4UkNvcm5lcik7XG4gICAgICAgIGNvbnN0IHhSTWF4ID1cbiAgICAgICAgICAgIE1hdGgubWluKGNvbnZJbmZvLmluSGVpZ2h0LCBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgKyB4UkNvcm5lcik7XG4gICAgICAgIGZvciAobGV0IHlDID0gMDsgeUMgPCBjb252SW5mby5vdXRXaWR0aDsgKyt5Qykge1xuICAgICAgICAgIGNvbnN0IHhDQ29ybmVyID0geUMgKiBzdHJpZGVXaWR0aCAtIHBhZExlZnQ7XG4gICAgICAgICAgbGV0IHhDTWluID0geENDb3JuZXI7XG4gICAgICAgICAgd2hpbGUgKHhDTWluIDwgMCkge1xuICAgICAgICAgICAgeENNaW4gKz0gZGlsYXRpb25XaWR0aDtcbiAgICAgICAgICB9XG4gICAgICAgICAgY29uc3QgeENNYXggPVxuICAgICAgICAgICAgICBNYXRoLm1pbihjb252SW5mby5pbldpZHRoLCBlZmZlY3RpdmVGaWx0ZXJXaWR0aCArIHhDQ29ybmVyKTtcbiAgICAgICAgICBsZXQgbWF4VmFsdWUgPSBOdW1iZXIuTkVHQVRJVkVfSU5GSU5JVFk7XG4gICAgICAgICAgbGV0IG1heFBvc2l0aW9uID0gLTE7XG5cbiAgICAgICAgICBmb3IgKGxldCB4UiA9IHhSTWluOyB4UiA8IHhSTWF4OyB4UiArPSBkaWxhdGlvbkhlaWdodCkge1xuICAgICAgICAgICAgY29uc3Qgd1IgPSB4UiAtIHhSQ29ybmVyO1xuICAgICAgICAgICAgZm9yIChsZXQgeEMgPSB4Q01pbjsgeEMgPCB4Q01heDsgeEMgKz0gZGlsYXRpb25XaWR0aCkge1xuICAgICAgICAgICAgICBjb25zdCB3QyA9IHhDIC0geENDb3JuZXI7XG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geEJ1Zi5nZXQoYiwgeFIsIHhDLCBkKTtcbiAgICAgICAgICAgICAgaWYgKHBpeGVsID4gbWF4VmFsdWUpIHtcbiAgICAgICAgICAgICAgICBtYXhWYWx1ZSA9IHBpeGVsIGFzIG51bWJlcjtcbiAgICAgICAgICAgICAgICBpZiAoZmxhdHRlblBvc2l0aW9ucykge1xuICAgICAgICAgICAgICAgICAgbWF4UG9zaXRpb24gPSBpbmNsdWRlQmF0Y2hJbkluZGV4ID9cbiAgICAgICAgICAgICAgICAgICAgICAoKGIgKiBjb252SW5mby5pbkhlaWdodCArIHhSKSAqIGNvbnZJbmZvLmluV2lkdGggKyB4QykgKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29udkluZm8uaW5DaGFubmVscyArXG4gICAgICAgICAgICAgICAgICAgICAgICAgIGQgOlxuICAgICAgICAgICAgICAgICAgICAgICh4UiAqIGNvbnZJbmZvLmluV2lkdGggKyB4QykgKiBjb252SW5mby5pbkNoYW5uZWxzICsgZDtcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgbWF4UG9zaXRpb24gPSB3UiAqIGVmZmVjdGl2ZUZpbHRlcldpZHRoICsgd0M7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIG1heFBvc2l0aW9ucy5zZXQobWF4UG9zaXRpb24sIGIsIHlSLCB5QywgZCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cbiAgcmV0dXJuIG1heFBvc2l0aW9ucztcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHBvb2wzZChcbiAgICB4VmFsdWVzOiBUeXBlZEFycmF5LCB4U2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUsIHN0cmlkZXM6IG51bWJlcltdLFxuICAgIGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjNESW5mbyxcbiAgICBwb29sVHlwZTogJ21heCd8J2F2ZycpOiBUZW5zb3JCdWZmZXI8UmFuaywgRGF0YVR5cGU+IHtcbiAgY29uc3Qgc3RyaWRlRGVwdGggPSBjb252SW5mby5zdHJpZGVEZXB0aDtcbiAgY29uc3Qgc3RyaWRlSGVpZ2h0ID0gY29udkluZm8uc3RyaWRlSGVpZ2h0O1xuICBjb25zdCBzdHJpZGVXaWR0aCA9IGNvbnZJbmZvLnN0cmlkZVdpZHRoO1xuICBjb25zdCBkaWxhdGlvbkRlcHRoID0gY29udkluZm8uZGlsYXRpb25EZXB0aDtcbiAgY29uc3QgZGlsYXRpb25IZWlnaHQgPSBjb252SW5mby5kaWxhdGlvbkhlaWdodDtcbiAgY29uc3QgZGlsYXRpb25XaWR0aCA9IGNvbnZJbmZvLmRpbGF0aW9uV2lkdGg7XG4gIGNvbnN0IGVmZmVjdGl2ZUZpbHRlckRlcHRoID0gY29udkluZm8uZWZmZWN0aXZlRmlsdGVyRGVwdGg7XG4gIGNvbnN0IGVmZmVjdGl2ZUZpbHRlckhlaWdodCA9IGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlckhlaWdodDtcbiAgY29uc3QgZWZmZWN0aXZlRmlsdGVyV2lkdGggPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJXaWR0aDtcbiAgY29uc3QgcGFkRnJvbnQgPSBjb252SW5mby5wYWRJbmZvLmZyb250O1xuICBjb25zdCBwYWRUb3AgPSBjb252SW5mby5wYWRJbmZvLnRvcDtcbiAgY29uc3QgcGFkTGVmdCA9IGNvbnZJbmZvLnBhZEluZm8ubGVmdDtcblxuICBjb25zdCBpbml0aWFsVmFsdWUgPVxuICAgICAgKHBvb2xUeXBlID09PSAnbWF4JyA/IE51bWJlci5ORUdBVElWRV9JTkZJTklUWSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgTnVtYmVyLlBPU0lUSVZFX0lORklOSVRZKTtcblxuICBjb25zdCBvdXRwdXQgPSBidWZmZXIoY29udkluZm8ub3V0U2hhcGUsIGR0eXBlKTtcbiAgY29uc3Qgb3V0cHV0VmFscyA9IG91dHB1dC52YWx1ZXM7XG5cbiAgY29uc3Qgb3V0cHV0QmF0Y2hTdHJpZGVzID0gY29udkluZm8ub3V0U2hhcGVbMV0gKiBjb252SW5mby5vdXRTaGFwZVsyXSAqXG4gICAgICBjb252SW5mby5vdXRTaGFwZVszXSAqIGNvbnZJbmZvLm91dFNoYXBlWzRdO1xuICBjb25zdCBvdXRwdXREZXB0aFN0cmlkZXMgPVxuICAgICAgY29udkluZm8ub3V0U2hhcGVbMl0gKiBjb252SW5mby5vdXRTaGFwZVszXSAqIGNvbnZJbmZvLm91dFNoYXBlWzRdO1xuICBjb25zdCBvdXRwdXRSb3dTdHJpZGVzID0gY29udkluZm8ub3V0U2hhcGVbM10gKiBjb252SW5mby5vdXRTaGFwZVs0XTtcbiAgY29uc3Qgb3V0cHV0Q29sU3RyaWRlcyA9IGNvbnZJbmZvLm91dFNoYXBlWzRdO1xuXG4gIGZvciAobGV0IGJhdGNoID0gMDsgYmF0Y2ggPCBjb252SW5mby5iYXRjaFNpemU7ICsrYmF0Y2gpIHtcbiAgICBjb25zdCBvdXRwdXRCYXRjaE9mZnNldCA9IGJhdGNoICogb3V0cHV0QmF0Y2hTdHJpZGVzO1xuICAgIGNvbnN0IGlucHV0QmF0Y2hPZmZzZXQgPSBiYXRjaCAqIHN0cmlkZXNbMF07XG4gICAgZm9yIChsZXQgY2hhbm5lbCA9IDA7IGNoYW5uZWwgPCBjb252SW5mby5pbkNoYW5uZWxzOyArK2NoYW5uZWwpIHtcbiAgICAgIGZvciAobGV0IHlEZXB0aCA9IDA7IHlEZXB0aCA8IGNvbnZJbmZvLm91dERlcHRoOyArK3lEZXB0aCkge1xuICAgICAgICBjb25zdCB4RGVwdGhDb3JuZXIgPSB5RGVwdGggKiBzdHJpZGVEZXB0aCAtIHBhZEZyb250O1xuICAgICAgICBsZXQgeERlcHRoTWluID0geERlcHRoQ29ybmVyO1xuICAgICAgICB3aGlsZSAoeERlcHRoTWluIDwgMCkge1xuICAgICAgICAgIHhEZXB0aE1pbiArPSBkaWxhdGlvbkRlcHRoO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IHhEZXB0aE1heCA9XG4gICAgICAgICAgICBNYXRoLm1pbihjb252SW5mby5pbkRlcHRoLCBlZmZlY3RpdmVGaWx0ZXJEZXB0aCArIHhEZXB0aENvcm5lcik7XG4gICAgICAgIGNvbnN0IG91dHB1dERlcHRoT2Zmc2V0ID1cbiAgICAgICAgICAgIG91dHB1dEJhdGNoT2Zmc2V0ICsgeURlcHRoICogb3V0cHV0RGVwdGhTdHJpZGVzO1xuICAgICAgICBmb3IgKGxldCB5Um93ID0gMDsgeVJvdyA8IGNvbnZJbmZvLm91dEhlaWdodDsgKyt5Um93KSB7XG4gICAgICAgICAgY29uc3QgeFJvd0Nvcm5lciA9IHlSb3cgKiBzdHJpZGVIZWlnaHQgLSBwYWRUb3A7XG4gICAgICAgICAgbGV0IHhSb3dNaW4gPSB4Um93Q29ybmVyO1xuICAgICAgICAgIHdoaWxlICh4Um93TWluIDwgMCkge1xuICAgICAgICAgICAgeFJvd01pbiArPSBkaWxhdGlvbkhlaWdodDtcbiAgICAgICAgICB9XG4gICAgICAgICAgY29uc3QgeFJvd01heCA9XG4gICAgICAgICAgICAgIE1hdGgubWluKGNvbnZJbmZvLmluSGVpZ2h0LCBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgKyB4Um93Q29ybmVyKTtcbiAgICAgICAgICBjb25zdCBvdXRwdXRSb3dPZmZzZXQgPSBvdXRwdXREZXB0aE9mZnNldCArIHlSb3cgKiBvdXRwdXRSb3dTdHJpZGVzO1xuICAgICAgICAgIGZvciAobGV0IHlDb2wgPSAwOyB5Q29sIDwgY29udkluZm8ub3V0V2lkdGg7ICsreUNvbCkge1xuICAgICAgICAgICAgY29uc3QgeENvbENvcm5lciA9IHlDb2wgKiBzdHJpZGVXaWR0aCAtIHBhZExlZnQ7XG4gICAgICAgICAgICBsZXQgeENvbE1pbiA9IHhDb2xDb3JuZXI7XG4gICAgICAgICAgICB3aGlsZSAoeENvbE1pbiA8IDApIHtcbiAgICAgICAgICAgICAgeENvbE1pbiArPSBkaWxhdGlvbldpZHRoO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgY29uc3QgeENvbE1heCA9XG4gICAgICAgICAgICAgICAgTWF0aC5taW4oY29udkluZm8uaW5XaWR0aCwgZWZmZWN0aXZlRmlsdGVyV2lkdGggKyB4Q29sQ29ybmVyKTtcbiAgICAgICAgICAgIC8vIFNoYWRlciBjb2RlIGJlZ2luc1xuICAgICAgICAgICAgY29uc3Qgb3V0cHV0Q29sT2Zmc2V0ID0gb3V0cHV0Um93T2Zmc2V0ICsgeUNvbCAqIG91dHB1dENvbFN0cmlkZXM7XG4gICAgICAgICAgICBsZXQgbWluTWF4VmFsdWUgPSBpbml0aWFsVmFsdWU7XG4gICAgICAgICAgICBsZXQgYXZnVmFsdWUgPSAwO1xuICAgICAgICAgICAgbGV0IGNvdW50ID0gMDtcbiAgICAgICAgICAgIGZvciAobGV0IHhEZXB0aCA9IHhEZXB0aE1pbjsgeERlcHRoIDwgeERlcHRoTWF4O1xuICAgICAgICAgICAgICAgICB4RGVwdGggKz0gZGlsYXRpb25EZXB0aCkge1xuICAgICAgICAgICAgICBjb25zdCB4RGVwdGhPZmZzZXQgPSBpbnB1dEJhdGNoT2Zmc2V0ICsgeERlcHRoICogc3RyaWRlc1sxXTtcbiAgICAgICAgICAgICAgZm9yIChsZXQgeFJvdyA9IHhSb3dNaW47IHhSb3cgPCB4Um93TWF4OyB4Um93ICs9IGRpbGF0aW9uSGVpZ2h0KSB7XG4gICAgICAgICAgICAgICAgY29uc3QgeFJvd09mZnNldCA9IHhEZXB0aE9mZnNldCArIHhSb3cgKiBzdHJpZGVzWzJdO1xuICAgICAgICAgICAgICAgIGZvciAobGV0IHhDb2wgPSB4Q29sTWluOyB4Q29sIDwgeENvbE1heDtcbiAgICAgICAgICAgICAgICAgICAgIHhDb2wgKz0gZGlsYXRpb25XaWR0aCkge1xuICAgICAgICAgICAgICAgICAgY29uc3QgeENvbE9mZnNldCA9IHhSb3dPZmZzZXQgKyB4Q29sICogc3RyaWRlc1szXTtcbiAgICAgICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geFZhbHVlc1t4Q29sT2Zmc2V0ICsgY2hhbm5lbF07XG4gICAgICAgICAgICAgICAgICBpZiAoKHBvb2xUeXBlID09PSAnbWF4JyAmJiBwaXhlbCA+IG1pbk1heFZhbHVlKSkge1xuICAgICAgICAgICAgICAgICAgICBtaW5NYXhWYWx1ZSA9IHBpeGVsO1xuICAgICAgICAgICAgICAgICAgfSBlbHNlIGlmIChwb29sVHlwZSA9PT0gJ2F2ZycpIHtcbiAgICAgICAgICAgICAgICAgICAgYXZnVmFsdWUgKz0gcGl4ZWw7XG4gICAgICAgICAgICAgICAgICAgIGNvdW50Kys7XG4gICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICBpZiAoaXNOYU4obWluTWF4VmFsdWUpKSB7XG4gICAgICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBpZiAoaXNOYU4obWluTWF4VmFsdWUpKSB7XG4gICAgICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgaWYgKGlzTmFOKG1pbk1heFZhbHVlKSkge1xuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBjb25zdCBvdXRwdXRPZmZzZXQgPSBvdXRwdXRDb2xPZmZzZXQgKyBjaGFubmVsO1xuICAgICAgICAgICAgb3V0cHV0VmFsc1tvdXRwdXRPZmZzZXRdID0gcG9vbFR5cGUgPT09ICdhdmcnID9cbiAgICAgICAgICAgICAgICBhdmdWYWx1ZSAvIE1hdGgubWF4KGNvdW50LCAxKSA6XG4gICAgICAgICAgICAgICAgbWluTWF4VmFsdWU7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIG91dHB1dDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIG1heFBvb2wzZFBvc2l0aW9ucyhcbiAgICB4QnVmOiBUZW5zb3JCdWZmZXI8UmFuaywgRGF0YVR5cGU+LFxuICAgIGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjNESW5mbyk6IFRlbnNvckJ1ZmZlcjxSYW5rLCBEYXRhVHlwZT4ge1xuICBjb25zdCBtYXhQb3NpdGlvbnMgPSBidWZmZXIoY29udkluZm8ub3V0U2hhcGUsICdpbnQzMicpO1xuICBjb25zdCBzdHJpZGVEZXB0aCA9IGNvbnZJbmZvLnN0cmlkZURlcHRoO1xuICBjb25zdCBzdHJpZGVIZWlnaHQgPSBjb252SW5mby5zdHJpZGVIZWlnaHQ7XG4gIGNvbnN0IHN0cmlkZVdpZHRoID0gY29udkluZm8uc3RyaWRlV2lkdGg7XG4gIGNvbnN0IGRpbGF0aW9uRGVwdGggPSBjb252SW5mby5kaWxhdGlvbkRlcHRoO1xuICBjb25zdCBkaWxhdGlvbkhlaWdodCA9IGNvbnZJbmZvLmRpbGF0aW9uSGVpZ2h0O1xuICBjb25zdCBkaWxhdGlvbldpZHRoID0gY29udkluZm8uZGlsYXRpb25XaWR0aDtcbiAgY29uc3QgZWZmZWN0aXZlRmlsdGVyRGVwdGggPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJEZXB0aDtcbiAgY29uc3QgZWZmZWN0aXZlRmlsdGVySGVpZ2h0ID0gY29udkluZm8uZWZmZWN0aXZlRmlsdGVySGVpZ2h0O1xuICBjb25zdCBlZmZlY3RpdmVGaWx0ZXJXaWR0aCA9IGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlcldpZHRoO1xuICBjb25zdCBwYWRGcm9udCA9IGNvbnZJbmZvLnBhZEluZm8uZnJvbnQ7XG4gIGNvbnN0IHBhZFRvcCA9IGNvbnZJbmZvLnBhZEluZm8udG9wO1xuICBjb25zdCBwYWRMZWZ0ID0gY29udkluZm8ucGFkSW5mby5sZWZ0O1xuXG4gIGZvciAobGV0IGJhdGNoID0gMDsgYmF0Y2ggPCBjb252SW5mby5iYXRjaFNpemU7ICsrYmF0Y2gpIHtcbiAgICBmb3IgKGxldCBjaGFubmVsID0gMDsgY2hhbm5lbCA8IGNvbnZJbmZvLmluQ2hhbm5lbHM7ICsrY2hhbm5lbCkge1xuICAgICAgZm9yIChsZXQgeURlcHRoID0gMDsgeURlcHRoIDwgY29udkluZm8ub3V0RGVwdGg7ICsreURlcHRoKSB7XG4gICAgICAgIGNvbnN0IHhEZXB0aENvcm5lciA9IHlEZXB0aCAqIHN0cmlkZURlcHRoIC0gcGFkRnJvbnQ7XG4gICAgICAgIGxldCB4RGVwdGhNaW4gPSB4RGVwdGhDb3JuZXI7XG4gICAgICAgIHdoaWxlICh4RGVwdGhNaW4gPCAwKSB7XG4gICAgICAgICAgeERlcHRoTWluICs9IGRpbGF0aW9uRGVwdGg7XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgeERlcHRoTWF4ID1cbiAgICAgICAgICAgIE1hdGgubWluKGNvbnZJbmZvLmluRGVwdGgsIGVmZmVjdGl2ZUZpbHRlckRlcHRoICsgeERlcHRoQ29ybmVyKTtcbiAgICAgICAgZm9yIChsZXQgeVJvdyA9IDA7IHlSb3cgPCBjb252SW5mby5vdXRIZWlnaHQ7ICsreVJvdykge1xuICAgICAgICAgIGNvbnN0IHhSb3dDb3JuZXIgPSB5Um93ICogc3RyaWRlSGVpZ2h0IC0gcGFkVG9wO1xuICAgICAgICAgIGxldCB4Um93TWluID0geFJvd0Nvcm5lcjtcbiAgICAgICAgICB3aGlsZSAoeFJvd01pbiA8IDApIHtcbiAgICAgICAgICAgIHhSb3dNaW4gKz0gZGlsYXRpb25IZWlnaHQ7XG4gICAgICAgICAgfVxuICAgICAgICAgIGNvbnN0IHhSb3dNYXggPVxuICAgICAgICAgICAgICBNYXRoLm1pbihjb252SW5mby5pbkhlaWdodCwgZWZmZWN0aXZlRmlsdGVySGVpZ2h0ICsgeFJvd0Nvcm5lcik7XG4gICAgICAgICAgZm9yIChsZXQgeUNvbCA9IDA7IHlDb2wgPCBjb252SW5mby5vdXRXaWR0aDsgKyt5Q29sKSB7XG4gICAgICAgICAgICBjb25zdCB4Q29sQ29ybmVyID0geUNvbCAqIHN0cmlkZVdpZHRoIC0gcGFkTGVmdDtcbiAgICAgICAgICAgIGxldCB4Q29sTWluID0geENvbENvcm5lcjtcbiAgICAgICAgICAgIHdoaWxlICh4Q29sTWluIDwgMCkge1xuICAgICAgICAgICAgICB4Q29sTWluICs9IGRpbGF0aW9uV2lkdGg7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBjb25zdCB4Q29sTWF4ID1cbiAgICAgICAgICAgICAgICBNYXRoLm1pbihjb252SW5mby5pbldpZHRoLCBlZmZlY3RpdmVGaWx0ZXJXaWR0aCArIHhDb2xDb3JuZXIpO1xuXG4gICAgICAgICAgICAvLyBTaGFkZXIgY29kZSBiZWdpbnNcbiAgICAgICAgICAgIGxldCBtYXhWYWx1ZSA9IE51bWJlci5ORUdBVElWRV9JTkZJTklUWTtcbiAgICAgICAgICAgIGxldCBtYXhQb3NpdGlvbiA9IC0xO1xuXG4gICAgICAgICAgICBmb3IgKGxldCB4RGVwdGggPSB4RGVwdGhNaW47IHhEZXB0aCA8IHhEZXB0aE1heDtcbiAgICAgICAgICAgICAgICAgeERlcHRoICs9IGRpbGF0aW9uRGVwdGgpIHtcbiAgICAgICAgICAgICAgY29uc3Qgd0RlcHRoID0geERlcHRoIC0geERlcHRoQ29ybmVyO1xuICAgICAgICAgICAgICBmb3IgKGxldCB4Um93ID0geFJvd01pbjsgeFJvdyA8IHhSb3dNYXg7IHhSb3cgKz0gZGlsYXRpb25IZWlnaHQpIHtcbiAgICAgICAgICAgICAgICBjb25zdCB3Um93ID0geFJvdyAtIHhSb3dDb3JuZXI7XG4gICAgICAgICAgICAgICAgZm9yIChsZXQgeENvbCA9IHhDb2xNaW47IHhDb2wgPCB4Q29sTWF4O1xuICAgICAgICAgICAgICAgICAgICAgeENvbCArPSBkaWxhdGlvbldpZHRoKSB7XG4gICAgICAgICAgICAgICAgICBjb25zdCB3Q29sID0geENvbCAtIHhDb2xDb3JuZXI7XG4gICAgICAgICAgICAgICAgICBjb25zdCBwaXhlbCA9IHhCdWYuZ2V0KGJhdGNoLCB4RGVwdGgsIHhSb3csIHhDb2wsIGNoYW5uZWwpO1xuICAgICAgICAgICAgICAgICAgaWYgKHBpeGVsID49IG1heFZhbHVlKSB7XG4gICAgICAgICAgICAgICAgICAgIG1heFZhbHVlID0gcGl4ZWwgYXMgbnVtYmVyO1xuICAgICAgICAgICAgICAgICAgICBtYXhQb3NpdGlvbiA9XG4gICAgICAgICAgICAgICAgICAgICAgICB3RGVwdGggKiBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgKiBlZmZlY3RpdmVGaWx0ZXJXaWR0aCArXG4gICAgICAgICAgICAgICAgICAgICAgICB3Um93ICogZWZmZWN0aXZlRmlsdGVySGVpZ2h0ICsgd0NvbDtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgbWF4UG9zaXRpb25zLnNldChtYXhQb3NpdGlvbiwgYmF0Y2gsIHlEZXB0aCwgeVJvdywgeUNvbCwgY2hhbm5lbCk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIG1heFBvc2l0aW9ucztcbn1cbiJdfQ==