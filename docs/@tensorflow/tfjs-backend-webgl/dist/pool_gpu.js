/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
export class Pool2DProgram {
    constructor(convInfo, poolType, computePositions, flattenPositions = false, includeBatchInIndex = false) {
        this.variableNames = ['x'];
        if (poolType === 'avg' && computePositions) {
            throw new Error('Cannot compute positions for average pool.');
        }
        const filterWidth = convInfo.filterWidth;
        const strideHeight = convInfo.strideHeight;
        const strideWidth = convInfo.strideWidth;
        const dilationHeight = convInfo.dilationHeight;
        const dilationWidth = convInfo.dilationWidth;
        const effectiveFilterHeight = convInfo.effectiveFilterHeight;
        const effectiveFilterWidth = convInfo.effectiveFilterWidth;
        const padTop = convInfo.padInfo.top;
        const padLeft = convInfo.padInfo.left;
        this.outputShape = convInfo.outShape;
        const isAvgPool = poolType === 'avg';
        const batchFlattenPositionStr = `((batch  * ${convInfo.inHeight} + xR) * ${convInfo.inWidth} + xC) * ${convInfo.inChannels} + d`;
        const flattenPositionStr = `(xR * ${convInfo.inWidth} + xC) * ${convInfo.inChannels} + d`;
        let initializationValue = '0.0';
        if (!isAvgPool) {
            // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
            initializationValue = '-1.0 / 1e-20';
        }
        if (computePositions) {
            const compareOp = '>=';
            this.userCode = `
        const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
        const ivec2 pads = ivec2(${padTop}, ${padLeft});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${effectiveFilterHeight};
              wR += ${dilationHeight}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${effectiveFilterWidth};
                wC += ${dilationWidth}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${convInfo.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value ${compareOp} currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${flattenPositions ? (includeBatchInIndex ? batchFlattenPositionStr :
                flattenPositionStr) :
                `wR * ${effectiveFilterWidth} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;
            return;
        }
        const compareOp = 'max';
        let returnValue = `${poolType}(${poolType}(${poolType}(` +
            'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
        if (poolType === 'avg') {
            returnValue = `avgValue / max(count, 1.0)`;
        }
        const filterWidthNearestVec4 = Math.floor(filterWidth / 4) * 4;
        const filterWidthVec4Remainder = filterWidth % 4;
        const updateSnippet = `
      if (${isAvgPool}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${compareOp}(values, minMaxValue);
      }
    `;
        this.userCode = `
      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});
      const float initializationValue = ${initializationValue};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${convInfo.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${initializationValue});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${effectiveFilterHeight};
            wR += ${dilationHeight}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${convInfo.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${filterWidthNearestVec4}; wC += 4) {
            int xC = xCCorner + wC * ${dilationWidth};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${dilationWidth}, d),
              getValue(batch, xR, xC + 2 * ${dilationWidth}, d),
              getValue(batch, xR, xC + 3 * ${dilationWidth}, d)
            );

            ${updateSnippet}
          }

          int xC = xCCorner + ${filterWidthNearestVec4};
          if (${filterWidthVec4Remainder === 1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${updateSnippet}
          } else if (${filterWidthVec4Remainder === 2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${dilationWidth}, d),
              initializationValue,
              initializationValue
            );

            ${updateSnippet}
          } else if (${filterWidthVec4Remainder === 3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${dilationWidth}, d),
              getValue(batch, xR, xC + 2 * ${dilationWidth}, d),
              initializationValue
            );

            ${updateSnippet}
          }
        }
        setOutput(${returnValue});
      }
    `;
    }
}
export class Pool3DProgram {
    constructor(convInfo, poolType, computePositions, flattenPositions = false, includeBatchInIndex = false) {
        this.variableNames = ['x'];
        if (poolType === 'avg' && computePositions) {
            throw new Error('Cannot compute positions for average pool.');
        }
        const filterWidth = convInfo.filterWidth;
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
        this.outputShape = convInfo.outShape;
        const isAvgPool = poolType === 'avg';
        let initializationValue = '0.0';
        if (!isAvgPool) {
            // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
            initializationValue = '-1.0 / 1e-20';
        }
        if (computePositions) {
            const compareOp = '>=';
            this.userCode = `
        const ivec3 strides =
            ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
        const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${effectiveFilterDepth};
              wD += ${dilationDepth}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${convInfo.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${effectiveFilterHeight};
                wR += ${dilationHeight}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${convInfo.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${effectiveFilterWidth};
                  wC += ${dilationWidth}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${convInfo.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value ${compareOp} currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${flattenPositions ?
                (includeBatchInIndex ?
                    `(((batch * ${convInfo.inDepth} + xD) * ${convInfo.inHeight} + xR) * ${convInfo.inWidth} + xC) * ${convInfo.inChannels} + ch` :
                    `((xD * ${convInfo.inHeight} + xR) * ${convInfo.inWidth} + xC) * ${convInfo.inChannels} + ch`) :
                `wD * ${effectiveFilterHeight} * ${effectiveFilterWidth} +
                      wR * ${effectiveFilterWidth} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;
            return;
        }
        const compareOp = 'max';
        let returnValue = `${poolType}(${poolType}(${poolType}(` +
            'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
        if (poolType === 'avg') {
            // Use `max(count, 1.0)` instead of `count` in case count === 0.0.
            // If count === 0.0, `avgValue` is always 0.0 and we change `count`'s
            // value to avoid dividing zero.
            returnValue = `avgValue / max(count, 1.0)`;
        }
        const filterWidthNearestVec4 = Math.floor(filterWidth / 4) * 4;
        const filterWidthVec4Remainder = filterWidth % 4;
        const updateSnippet = `
      if (${isAvgPool}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${compareOp}(values, minMaxValue);
      }
    `;
        this.userCode = `
      const ivec3 strides =
        ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
      const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});
      const float initializationValue = ${initializationValue};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${convInfo.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${initializationValue});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${effectiveFilterDepth};
            wD += ${dilationDepth}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${convInfo.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${effectiveFilterHeight};
            wR += ${dilationHeight}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${filterWidthNearestVec4}; wC += 4) {
              int xC = xCCorner + wC * ${dilationWidth};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${dilationWidth}, ch),
                getValue(batch, xD, xR, xC + 2 * ${dilationWidth}, ch),
                getValue(batch, xD, xR, xC + 3 * ${dilationWidth}, ch)
              );

              ${updateSnippet}
            }

            int xC = xCCorner + ${filterWidthNearestVec4};
            if (${filterWidthVec4Remainder === 1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${updateSnippet}
            } else if (${filterWidthVec4Remainder === 2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${dilationWidth}, ch),
                initializationValue,
                initializationValue
              );

              ${updateSnippet}
            } else if (${filterWidthVec4Remainder === 3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${dilationWidth}, ch),
                getValue(batch, xD, xR, xC + 2 * ${dilationWidth}, ch),
                initializationValue
              );

              ${updateSnippet}
            }
          }
        }
        setOutput(${returnValue});
      }
    `;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9vbF9ncHUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ2wvc3JjL3Bvb2xfZ3B1LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUtILE1BQU0sT0FBTyxhQUFhO0lBS3hCLFlBQ0ksUUFBaUMsRUFBRSxRQUFxQixFQUN4RCxnQkFBeUIsRUFBRSxnQkFBZ0IsR0FBRyxLQUFLLEVBQ25ELG1CQUFtQixHQUFHLEtBQUs7UUFQL0Isa0JBQWEsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBUXBCLElBQUksUUFBUSxLQUFLLEtBQUssSUFBSSxnQkFBZ0IsRUFBRTtZQUMxQyxNQUFNLElBQUksS0FBSyxDQUFDLDRDQUE0QyxDQUFDLENBQUM7U0FDL0Q7UUFFRCxNQUFNLFdBQVcsR0FBRyxRQUFRLENBQUMsV0FBVyxDQUFDO1FBQ3pDLE1BQU0sWUFBWSxHQUFHLFFBQVEsQ0FBQyxZQUFZLENBQUM7UUFDM0MsTUFBTSxXQUFXLEdBQUcsUUFBUSxDQUFDLFdBQVcsQ0FBQztRQUN6QyxNQUFNLGNBQWMsR0FBRyxRQUFRLENBQUMsY0FBYyxDQUFDO1FBQy9DLE1BQU0sYUFBYSxHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUM7UUFDN0MsTUFBTSxxQkFBcUIsR0FBRyxRQUFRLENBQUMscUJBQXFCLENBQUM7UUFDN0QsTUFBTSxvQkFBb0IsR0FBRyxRQUFRLENBQUMsb0JBQW9CLENBQUM7UUFFM0QsTUFBTSxNQUFNLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUM7UUFDcEMsTUFBTSxPQUFPLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUM7UUFDdEMsSUFBSSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUMsUUFBUSxDQUFDO1FBRXJDLE1BQU0sU0FBUyxHQUFHLFFBQVEsS0FBSyxLQUFLLENBQUM7UUFDckMsTUFBTSx1QkFBdUIsR0FB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Q0FDM0IsUUFBUSxLQUFLLE1BQU0sS0FBSyxPQUFPOzBDQUN0QixtQkFBbUI7Ozs7Ozs4QkFNL0IsUUFBUSxDQUFDLE9BQU87Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7a0NBbUJaLG1CQUFtQjs7OztnQ0FJckIsb0JBQW9CO29CQUNoQyxhQUFhOzs7Z0NBR0QsUUFBUSxDQUFDLE9BQU87Ozs7a0NBSWQscUJBQXFCO29CQUNuQyxjQUFjOzs7a0NBR0EsUUFBUSxDQUFDLFFBQVE7Ozs7b0NBSWYsc0JBQXNCO3lDQUNqQixhQUFhOzs7OytDQUlQLGFBQWE7bURBQ1QsYUFBYTttREFDYixhQUFhOzs7Z0JBR2hELGFBQWE7OztrQ0FHSyxzQkFBc0I7a0JBQ3RDLHdCQUF3QixLQUFLLENBQUM7Ozs7Ozs7O2dCQVFoQyxhQUFhO3lCQUNKLHdCQUF3QixLQUFLLENBQUM7OzsrQ0FHUixhQUFhOzs7OztnQkFLNUMsYUFBYTt5QkFDSix3QkFBd0IsS0FBSyxDQUFDOzs7K0NBR1IsYUFBYTttREFDVCxhQUFhOzs7O2dCQUloRCxhQUFhOzs7O29CQUlULFdBQVc7O0tBRTFCLENBQUM7SUFDSixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtHUEdQVVByb2dyYW19IGZyb20gJy4vZ3BncHVfbWF0aCc7XG5cbmV4cG9ydCBjbGFzcyBQb29sMkRQcm9ncmFtIGltcGxlbWVudHMgR1BHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCddO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHVzZXJDb2RlOiBzdHJpbmc7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBjb252SW5mbzogYmFja2VuZF91dGlsLkNvbnYyREluZm8sIHBvb2xUeXBlOiAnbWF4J3wnYXZnJyxcbiAgICAgIGNvbXB1dGVQb3NpdGlvbnM6IGJvb2xlYW4sIGZsYXR0ZW5Qb3NpdGlvbnMgPSBmYWxzZSxcbiAgICAgIGluY2x1ZGVCYXRjaEluSW5kZXggPSBmYWxzZSkge1xuICAgIGlmIChwb29sVHlwZSA9PT0gJ2F2ZycgJiYgY29tcHV0ZVBvc2l0aW9ucykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdDYW5ub3QgY29tcHV0ZSBwb3NpdGlvbnMgZm9yIGF2ZXJhZ2UgcG9vbC4nKTtcbiAgICB9XG5cbiAgICBjb25zdCBmaWx0ZXJXaWR0aCA9IGNvbnZJbmZvLmZpbHRlcldpZHRoO1xuICAgIGNvbnN0IHN0cmlkZUhlaWdodCA9IGNvbnZJbmZvLnN0cmlkZUhlaWdodDtcbiAgICBjb25zdCBzdHJpZGVXaWR0aCA9IGNvbnZJbmZvLnN0cmlkZVdpZHRoO1xuICAgIGNvbnN0IGRpbGF0aW9uSGVpZ2h0ID0gY29udkluZm8uZGlsYXRpb25IZWlnaHQ7XG4gICAgY29uc3QgZGlsYXRpb25XaWR0aCA9IGNvbnZJbmZvLmRpbGF0aW9uV2lkdGg7XG4gICAgY29uc3QgZWZmZWN0aXZlRmlsdGVySGVpZ2h0ID0gY29udkluZm8uZWZmZWN0aXZlRmlsdGVySGVpZ2h0O1xuICAgIGNvbnN0IGVmZmVjdGl2ZUZpbHRlcldpZHRoID0gY29udkluZm8uZWZmZWN0aXZlRmlsdGVyV2lkdGg7XG5cbiAgICBjb25zdCBwYWRUb3AgPSBjb252SW5mby5wYWRJbmZvLnRvcDtcbiAgICBjb25zdCBwYWRMZWZ0ID0gY29udkluZm8ucGFkSW5mby5sZWZ0O1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5vdXRTaGFwZTtcblxuICAgIGNvbnN0IGlzQXZnUG9vbCA9IHBvb2xUeXBlID09PSAnYXZnJztcbiAgICBjb25zdCBiYXRjaEZsYXR0ZW5Qb3NpdGlvblN0ciA9IGAoKGJhdGNoICAqICR7Y29udkluZm8uaW5IZWlnaHR9ICsgeFIpICogJHtcbiAgICAgICAgY29udkluZm8uaW5XaWR0aH0gKyB4QykgKiAke2NvbnZJbmZvLmluQ2hhbm5lbHN9ICsgZGA7XG4gICAgY29uc3QgZmxhdHRlblBvc2l0aW9uU3RyID1cbiAgICAgICAgYCh4UiAqICR7Y29udkluZm8uaW5XaWR0aH0gKyB4QykgKiAke2NvbnZJbmZvLmluQ2hhbm5lbHN9ICsgZGA7XG5cbiAgICBsZXQgaW5pdGlhbGl6YXRpb25WYWx1ZSA9ICcwLjAnO1xuICAgIGlmICghaXNBdmdQb29sKSB7XG4gICAgICAvLyBXZWJHTCBvbiBGaXJlZm94IExpbnV4IGNhbid0IGNvbXBpbGUgMS8wIHNvIHdlIGRvIDEvZXBzLlxuICAgICAgaW5pdGlhbGl6YXRpb25WYWx1ZSA9ICctMS4wIC8gMWUtMjAnO1xuICAgIH1cblxuICAgIGlmIChjb21wdXRlUG9zaXRpb25zKSB7XG4gICAgICBjb25zdCBjb21wYXJlT3AgPSAnPj0nO1xuXG4gICAgICB0aGlzLnVzZXJDb2RlID0gYFxuICAgICAgICBjb25zdCBpdmVjMiBzdHJpZGVzID0gaXZlYzIoJHtzdHJpZGVIZWlnaHR9LCAke3N0cmlkZVdpZHRofSk7XG4gICAgICAgIGNvbnN0IGl2ZWMyIHBhZHMgPSBpdmVjMigke3BhZFRvcH0sICR7cGFkTGVmdH0pO1xuXG4gICAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgICBpdmVjNCBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgICBpbnQgYmF0Y2ggPSBjb29yZHNbMF07XG4gICAgICAgICAgaW50IGQgPSBjb29yZHNbM107XG5cbiAgICAgICAgICBpdmVjMiB4UkNDb3JuZXIgPSBjb29yZHMueXogKiBzdHJpZGVzIC0gcGFkcztcbiAgICAgICAgICBpbnQgeFJDb3JuZXIgPSB4UkNDb3JuZXIueDtcbiAgICAgICAgICBpbnQgeENDb3JuZXIgPSB4UkNDb3JuZXIueTtcblxuICAgICAgICAgIC8vIG1heC9taW4geCg/LCA/LCBkKSB0byBnZXQgeSh5UiwgeUMsIGQpLlxuICAgICAgICAgIC8vID8gPSB0byBiZSBkZXRlcm1pbmVkXG4gICAgICAgICAgZmxvYXQgbWluTWF4VmFsdWUgPSAwLjA7XG4gICAgICAgICAgZmxvYXQgbWluTWF4VmFsdWVGb3VuZCA9IDAuMDtcbiAgICAgICAgICBpbnQgbWluTWF4UG9zaXRpb24gPSAwO1xuICAgICAgICAgIGZsb2F0IGF2Z1ZhbHVlID0gMC4wO1xuXG4gICAgICAgICAgZm9yIChpbnQgd1IgPSAwOyB3UiA8ICR7ZWZmZWN0aXZlRmlsdGVySGVpZ2h0fTtcbiAgICAgICAgICAgICAgd1IgKz0gJHtkaWxhdGlvbkhlaWdodH0pIHtcbiAgICAgICAgICAgIGludCB4UiA9IHhSQ29ybmVyICsgd1I7XG5cbiAgICAgICAgICAgIGlmICh4UiA8IDAgfHwgeFIgPj0gJHtjb252SW5mby5pbkhlaWdodH0pIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGZvciAoaW50IHdDID0gMDsgd0MgPCAke2VmZmVjdGl2ZUZpbHRlcldpZHRofTtcbiAgICAgICAgICAgICAgICB3QyArPSAke2RpbGF0aW9uV2lkdGh9KSB7XG4gICAgICAgICAgICAgIGludCB4QyA9IHhDQ29ybmVyICsgd0M7XG5cbiAgICAgICAgICAgICAgaWYgKHhDIDwgMCB8fCB4QyA+PSAke2NvbnZJbmZvLmluV2lkdGh9KSB7XG4gICAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICBmbG9hdCB2YWx1ZSA9IGdldFgoYmF0Y2gsIHhSLCB4QywgZCk7XG5cbiAgICAgICAgICAgICAgLy8gSWYgYSBtaW4gLyBtYXggdmFsdWUgaGFzIGFscmVhZHkgYmVlbiBmb3VuZCwgdXNlIGl0LiBJZiBub3QsXG4gICAgICAgICAgICAgIC8vIHVzZSB0aGUgY3VycmVudCB2YWx1ZS5cbiAgICAgICAgICAgICAgZmxvYXQgY3Vyck1pbk1heFZhbHVlID0gbWl4KFxuICAgICAgICAgICAgICAgICAgdmFsdWUsIG1pbk1heFZhbHVlLCBtaW5NYXhWYWx1ZUZvdW5kKTtcbiAgICAgICAgICAgICAgaWYgKHZhbHVlICR7Y29tcGFyZU9wfSBjdXJyTWluTWF4VmFsdWUpIHtcbiAgICAgICAgICAgICAgICBtaW5NYXhWYWx1ZSA9IHZhbHVlO1xuICAgICAgICAgICAgICAgIG1pbk1heFZhbHVlRm91bmQgPSAxLjA7XG4gICAgICAgICAgICAgICAgbWluTWF4UG9zaXRpb24gPSAke1xuICAgICAgICAgIGZsYXR0ZW5Qb3NpdGlvbnMgPyAoaW5jbHVkZUJhdGNoSW5JbmRleCA/IGJhdGNoRmxhdHRlblBvc2l0aW9uU3RyIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmbGF0dGVuUG9zaXRpb25TdHIpIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYHdSICogJHtlZmZlY3RpdmVGaWx0ZXJXaWR0aH0gKyB3Q2B9O1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIHNldE91dHB1dChmbG9hdChtaW5NYXhQb3NpdGlvbikpO1xuICAgICAgICB9XG4gICAgICBgO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IGNvbXBhcmVPcCA9ICdtYXgnO1xuXG4gICAgbGV0IHJldHVyblZhbHVlID0gYCR7cG9vbFR5cGV9KCR7cG9vbFR5cGV9KCR7cG9vbFR5cGV9KGAgK1xuICAgICAgICAnbWluTWF4VmFsdWVbMF0sIG1pbk1heFZhbHVlWzFdKSwgbWluTWF4VmFsdWVbMl0pLCBtaW5NYXhWYWx1ZVszXSknO1xuICAgIGlmIChwb29sVHlwZSA9PT0gJ2F2ZycpIHtcbiAgICAgIHJldHVyblZhbHVlID0gYGF2Z1ZhbHVlIC8gbWF4KGNvdW50LCAxLjApYDtcbiAgICB9XG5cbiAgICBjb25zdCBmaWx0ZXJXaWR0aE5lYXJlc3RWZWM0ID0gTWF0aC5mbG9vcihmaWx0ZXJXaWR0aCAvIDQpICogNDtcbiAgICBjb25zdCBmaWx0ZXJXaWR0aFZlYzRSZW1haW5kZXIgPSBmaWx0ZXJXaWR0aCAlIDQ7XG5cbiAgICBjb25zdCB1cGRhdGVTbmlwcGV0ID0gYFxuICAgICAgaWYgKCR7aXNBdmdQb29sfSkge1xuICAgICAgICBhdmdWYWx1ZSArPSBkb3QodmFsdWVzLCBvbmVzKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIG1pbk1heFZhbHVlID0gJHtjb21wYXJlT3B9KHZhbHVlcywgbWluTWF4VmFsdWUpO1xuICAgICAgfVxuICAgIGA7XG5cbiAgICB0aGlzLnVzZXJDb2RlID0gYFxuICAgICAgY29uc3QgaXZlYzIgc3RyaWRlcyA9IGl2ZWMyKCR7c3RyaWRlSGVpZ2h0fSwgJHtzdHJpZGVXaWR0aH0pO1xuICAgICAgY29uc3QgaXZlYzIgcGFkcyA9IGl2ZWMyKCR7cGFkVG9wfSwgJHtwYWRMZWZ0fSk7XG4gICAgICBjb25zdCBmbG9hdCBpbml0aWFsaXphdGlvblZhbHVlID0gJHtpbml0aWFsaXphdGlvblZhbHVlfTtcbiAgICAgIGNvbnN0IHZlYzQgb25lcyA9IHZlYzQoMS4wLCAxLjAsIDEuMCwgMS4wKTtcblxuICAgICAgZmxvYXQgY291bnQgPSAwLjA7XG5cbiAgICAgIGZsb2F0IGdldFZhbHVlKGludCBiYXRjaCwgaW50IHhSLCBpbnQgeEMsIGludCBkKSB7XG4gICAgICAgIGlmICh4QyA8IDAgfHwgeEMgPj0gJHtjb252SW5mby5pbldpZHRofSkge1xuICAgICAgICAgIHJldHVybiBpbml0aWFsaXphdGlvblZhbHVlO1xuICAgICAgICB9XG4gICAgICAgIGNvdW50ICs9IDEuMDtcbiAgICAgICAgcmV0dXJuIGdldFgoYmF0Y2gsIHhSLCB4QywgZCk7XG4gICAgICB9XG5cbiAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgaXZlYzQgY29vcmRzID0gZ2V0T3V0cHV0Q29vcmRzKCk7XG4gICAgICAgIGludCBiYXRjaCA9IGNvb3Jkc1swXTtcbiAgICAgICAgaW50IGQgPSBjb29yZHNbM107XG5cbiAgICAgICAgaXZlYzIgeFJDQ29ybmVyID0gY29vcmRzLnl6ICogc3RyaWRlcyAtIHBhZHM7XG4gICAgICAgIGludCB4UkNvcm5lciA9IHhSQ0Nvcm5lci54O1xuICAgICAgICBpbnQgeENDb3JuZXIgPSB4UkNDb3JuZXIueTtcblxuICAgICAgICAvLyBtYXgvbWluIHgoPywgPywgZCkgdG8gZ2V0IHkoeVIsIHlDLCBkKS5cbiAgICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWRcbiAgICAgICAgdmVjNCBtaW5NYXhWYWx1ZSA9IHZlYzQoJHtpbml0aWFsaXphdGlvblZhbHVlfSk7XG4gICAgICAgIGZsb2F0IGF2Z1ZhbHVlID0gMC4wO1xuICAgICAgICBjb3VudCA9IDAuMDtcblxuICAgICAgICBmb3IgKGludCB3UiA9IDA7IHdSIDwgJHtlZmZlY3RpdmVGaWx0ZXJIZWlnaHR9O1xuICAgICAgICAgICAgd1IgKz0gJHtkaWxhdGlvbkhlaWdodH0pIHtcbiAgICAgICAgICBpbnQgeFIgPSB4UkNvcm5lciArIHdSO1xuXG4gICAgICAgICAgaWYgKHhSIDwgMCB8fCB4UiA+PSAke2NvbnZJbmZvLmluSGVpZ2h0fSkge1xuICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgZm9yIChpbnQgd0MgPSAwOyB3QyA8ICR7ZmlsdGVyV2lkdGhOZWFyZXN0VmVjNH07IHdDICs9IDQpIHtcbiAgICAgICAgICAgIGludCB4QyA9IHhDQ29ybmVyICsgd0MgKiAke2RpbGF0aW9uV2lkdGh9O1xuXG4gICAgICAgICAgICB2ZWM0IHZhbHVlcyA9IHZlYzQoXG4gICAgICAgICAgICAgIGdldFZhbHVlKGJhdGNoLCB4UiwgeEMsIGQpLFxuICAgICAgICAgICAgICBnZXRWYWx1ZShiYXRjaCwgeFIsIHhDICsgJHtkaWxhdGlvbldpZHRofSwgZCksXG4gICAgICAgICAgICAgIGdldFZhbHVlKGJhdGNoLCB4UiwgeEMgKyAyICogJHtkaWxhdGlvbldpZHRofSwgZCksXG4gICAgICAgICAgICAgIGdldFZhbHVlKGJhdGNoLCB4UiwgeEMgKyAzICogJHtkaWxhdGlvbldpZHRofSwgZClcbiAgICAgICAgICAgICk7XG5cbiAgICAgICAgICAgICR7dXBkYXRlU25pcHBldH1cbiAgICAgICAgICB9XG5cbiAgICAgICAgICBpbnQgeEMgPSB4Q0Nvcm5lciArICR7ZmlsdGVyV2lkdGhOZWFyZXN0VmVjNH07XG4gICAgICAgICAgaWYgKCR7ZmlsdGVyV2lkdGhWZWM0UmVtYWluZGVyID09PSAxfSkge1xuICAgICAgICAgICAgdmVjNCB2YWx1ZXMgPSB2ZWM0KFxuICAgICAgICAgICAgICBnZXRWYWx1ZShiYXRjaCwgeFIsIHhDLCBkKSxcbiAgICAgICAgICAgICAgaW5pdGlhbGl6YXRpb25WYWx1ZSxcbiAgICAgICAgICAgICAgaW5pdGlhbGl6YXRpb25WYWx1ZSxcbiAgICAgICAgICAgICAgaW5pdGlhbGl6YXRpb25WYWx1ZVxuICAgICAgICAgICAgKTtcblxuICAgICAgICAgICAgJHt1cGRhdGVTbmlwcGV0fVxuICAgICAgICAgIH0gZWxzZSBpZiAoJHtmaWx0ZXJXaWR0aFZlYzRSZW1haW5kZXIgPT09IDJ9KSB7XG4gICAgICAgICAgICB2ZWM0IHZhbHVlcyA9IHZlYzQoXG4gICAgICAgICAgICAgIGdldFZhbHVlKGJhdGNoLCB4UiwgeEMsIGQpLFxuICAgICAgICAgICAgICBnZXRWYWx1ZShiYXRjaCwgeFIsIHhDICsgJHtkaWxhdGlvbldpZHRofSwgZCksXG4gICAgICAgICAgICAgIGluaXRpYWxpemF0aW9uVmFsdWUsXG4gICAgICAgICAgICAgIGluaXRpYWxpemF0aW9uVmFsdWVcbiAgICAgICAgICAgICk7XG5cbiAgICAgICAgICAgICR7dXBkYXRlU25pcHBldH1cbiAgICAgICAgICB9IGVsc2UgaWYgKCR7ZmlsdGVyV2lkdGhWZWM0UmVtYWluZGVyID09PSAzfSkge1xuICAgICAgICAgICAgdmVjNCB2YWx1ZXMgPSB2ZWM0KFxuICAgICAgICAgICAgICBnZXRWYWx1ZShiYXRjaCwgeFIsIHhDLCBkKSxcbiAgICAgICAgICAgICAgZ2V0VmFsdWUoYmF0Y2gsIHhSLCB4QyArICR7ZGlsYXRpb25XaWR0aH0sIGQpLFxuICAgICAgICAgICAgICBnZXRWYWx1ZShiYXRjaCwgeFIsIHhDICsgMiAqICR7ZGlsYXRpb25XaWR0aH0sIGQpLFxuICAgICAgICAgICAgICBpbml0aWFsaXphdGlvblZhbHVlXG4gICAgICAgICAgICApO1xuXG4gICAgICAgICAgICAke3VwZGF0ZVNuaXBwZXR9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHNldE91dHB1dCgke3JldHVyblZhbHVlfSk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgUG9vbDNEUHJvZ3JhbSBpbXBsZW1lbnRzIEdQR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnXTtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICB1c2VyQ29kZTogc3RyaW5nO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252M0RJbmZvLCBwb29sVHlwZTogJ21heCd8J2F2ZycsXG4gICAgICBjb21wdXRlUG9zaXRpb25zOiBib29sZWFuLCBmbGF0dGVuUG9zaXRpb25zID0gZmFsc2UsXG4gICAgICBpbmNsdWRlQmF0Y2hJbkluZGV4ID0gZmFsc2UpIHtcbiAgICBpZiAocG9vbFR5cGUgPT09ICdhdmcnICYmIGNvbXB1dGVQb3NpdGlvbnMpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignQ2Fubm90IGNvbXB1dGUgcG9zaXRpb25zIGZvciBhdmVyYWdlIHBvb2wuJyk7XG4gICAgfVxuXG4gICAgY29uc3QgZmlsdGVyV2lkdGggPSBjb252SW5mby5maWx0ZXJXaWR0aDtcbiAgICBjb25zdCBzdHJpZGVEZXB0aCA9IGNvbnZJbmZvLnN0cmlkZURlcHRoO1xuICAgIGNvbnN0IHN0cmlkZUhlaWdodCA9IGNvbnZJbmZvLnN0cmlkZUhlaWdodDtcbiAgICBjb25zdCBzdHJpZGVXaWR0aCA9IGNvbnZJbmZvLnN0cmlkZVdpZHRoO1xuICAgIGNvbnN0IGRpbGF0aW9uRGVwdGggPSBjb252SW5mby5kaWxhdGlvbkRlcHRoO1xuICAgIGNvbnN0IGRpbGF0aW9uSGVpZ2h0ID0gY29udkluZm8uZGlsYXRpb25IZWlnaHQ7XG4gICAgY29uc3QgZGlsYXRpb25XaWR0aCA9IGNvbnZJbmZvLmRpbGF0aW9uV2lkdGg7XG4gICAgY29uc3QgZWZmZWN0aXZlRmlsdGVyRGVwdGggPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJEZXB0aDtcbiAgICBjb25zdCBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJIZWlnaHQ7XG4gICAgY29uc3QgZWZmZWN0aXZlRmlsdGVyV2lkdGggPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJXaWR0aDtcblxuICAgIGNvbnN0IHBhZEZyb250ID0gY29udkluZm8ucGFkSW5mby5mcm9udDtcbiAgICBjb25zdCBwYWRUb3AgPSBjb252SW5mby5wYWRJbmZvLnRvcDtcbiAgICBjb25zdCBwYWRMZWZ0ID0gY29udkluZm8ucGFkSW5mby5sZWZ0O1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252SW5mby5vdXRTaGFwZTtcblxuICAgIGNvbnN0IGlzQXZnUG9vbCA9IHBvb2xUeXBlID09PSAnYXZnJztcblxuICAgIGxldCBpbml0aWFsaXphdGlvblZhbHVlID0gJzAuMCc7XG4gICAgaWYgKCFpc0F2Z1Bvb2wpIHtcbiAgICAgIC8vIFdlYkdMIG9uIEZpcmVmb3ggTGludXggY2FuJ3QgY29tcGlsZSAxLzAgc28gd2UgZG8gMS9lcHMuXG4gICAgICBpbml0aWFsaXphdGlvblZhbHVlID0gJy0xLjAgLyAxZS0yMCc7XG4gICAgfVxuXG4gICAgaWYgKGNvbXB1dGVQb3NpdGlvbnMpIHtcbiAgICAgIGNvbnN0IGNvbXBhcmVPcCA9ICc+PSc7XG5cbiAgICAgIHRoaXMudXNlckNvZGUgPSBgXG4gICAgICAgIGNvbnN0IGl2ZWMzIHN0cmlkZXMgPVxuICAgICAgICAgICAgaXZlYzMoJHtzdHJpZGVEZXB0aH0sICR7c3RyaWRlSGVpZ2h0fSwgJHtzdHJpZGVXaWR0aH0pO1xuICAgICAgICBjb25zdCBpdmVjMyBwYWRzID0gaXZlYzMoJHtwYWRGcm9udH0sICR7cGFkVG9wfSwgJHtwYWRMZWZ0fSk7XG5cbiAgICAgICAgdm9pZCBtYWluKCkge1xuICAgICAgICAgIGl2ZWM1IGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgICAgIGludCBiYXRjaCA9IGNvb3Jkcy54O1xuICAgICAgICAgIGludCBjaCA9IGNvb3Jkcy51O1xuXG4gICAgICAgICAgaXZlYzMgeENvcm5lciA9IGl2ZWMzKGNvb3Jkcy55LCBjb29yZHMueiwgY29vcmRzLncpICogc3RyaWRlcyAtIHBhZHM7XG4gICAgICAgICAgaW50IHhEQ29ybmVyID0geENvcm5lci54O1xuICAgICAgICAgIGludCB4UkNvcm5lciA9IHhDb3JuZXIueTtcbiAgICAgICAgICBpbnQgeENDb3JuZXIgPSB4Q29ybmVyLno7XG5cbiAgICAgICAgICAvLyBtYXgvbWluIHgoPywgPywgPywgY2gpIHRvIGdldCB5KHlELCB5UiwgeUMsIGNoKS5cbiAgICAgICAgICAvLyA/ID0gdG8gYmUgZGV0ZXJtaW5lZFxuICAgICAgICAgIGZsb2F0IG1pbk1heFZhbHVlID0gMC4wO1xuICAgICAgICAgIGZsb2F0IG1pbk1heFZhbHVlRm91bmQgPSAwLjA7XG4gICAgICAgICAgaW50IG1pbk1heFBvc2l0aW9uID0gMDtcblxuICAgICAgICAgIGZvciAoaW50IHdEID0gMDsgd0QgPCAke2VmZmVjdGl2ZUZpbHRlckRlcHRofTtcbiAgICAgICAgICAgICAgd0QgKz0gJHtkaWxhdGlvbkRlcHRofSkge1xuICAgICAgICAgICAgaW50IHhEID0geERDb3JuZXIgKyB3RDtcblxuICAgICAgICAgICAgaWYgKHhEIDwgMCB8fCB4RCA+PSAke2NvbnZJbmZvLmluRGVwdGh9KSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBmb3IgKGludCB3UiA9IDA7IHdSIDwgJHtlZmZlY3RpdmVGaWx0ZXJIZWlnaHR9O1xuICAgICAgICAgICAgICAgIHdSICs9ICR7ZGlsYXRpb25IZWlnaHR9KSB7XG4gICAgICAgICAgICAgIGludCB4UiA9IHhSQ29ybmVyICsgd1I7XG5cbiAgICAgICAgICAgICAgaWYgKHhSIDwgMCB8fCB4UiA+PSAke2NvbnZJbmZvLmluSGVpZ2h0fSkge1xuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgZm9yIChpbnQgd0MgPSAwOyB3QyA8ICR7ZWZmZWN0aXZlRmlsdGVyV2lkdGh9O1xuICAgICAgICAgICAgICAgICAgd0MgKz0gJHtkaWxhdGlvbldpZHRofSkge1xuICAgICAgICAgICAgICAgIGludCB4QyA9IHhDQ29ybmVyICsgd0M7XG5cbiAgICAgICAgICAgICAgICBpZiAoeEMgPCAwIHx8IHhDID49ICR7Y29udkluZm8uaW5XaWR0aH0pIHtcbiAgICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICAgIGZsb2F0IHZhbHVlID0gZ2V0WChiYXRjaCwgeEQsIHhSLCB4QywgY2gpO1xuXG4gICAgICAgICAgICAgICAgLy8gSWYgYSBtaW4gLyBtYXggdmFsdWUgaGFzIGFscmVhZHkgYmVlbiBmb3VuZCwgdXNlIGl0LiBJZiBub3QsXG4gICAgICAgICAgICAgICAgLy8gdXNlIHRoZSBjdXJyZW50IHZhbHVlLlxuICAgICAgICAgICAgICAgIGZsb2F0IGN1cnJNaW5NYXhWYWx1ZSA9IG1peChcbiAgICAgICAgICAgICAgICAgICAgdmFsdWUsIG1pbk1heFZhbHVlLCBtaW5NYXhWYWx1ZUZvdW5kKTtcbiAgICAgICAgICAgICAgICBpZiAodmFsdWUgJHtjb21wYXJlT3B9IGN1cnJNaW5NYXhWYWx1ZSkge1xuICAgICAgICAgICAgICAgICAgbWluTWF4VmFsdWUgPSB2YWx1ZTtcbiAgICAgICAgICAgICAgICAgIG1pbk1heFZhbHVlRm91bmQgPSAxLjA7XG4gICAgICAgICAgICAgICAgICBtaW5NYXhQb3NpdGlvbiA9ICR7XG4gICAgICAgICAgZmxhdHRlblBvc2l0aW9ucyA/XG4gICAgICAgICAgICAgIChpbmNsdWRlQmF0Y2hJbkluZGV4ID9cbiAgICAgICAgICAgICAgICAgICBgKCgoYmF0Y2ggKiAke2NvbnZJbmZvLmluRGVwdGh9ICsgeEQpICogJHtcbiAgICAgICAgICAgICAgICAgICAgICAgY29udkluZm8uaW5IZWlnaHR9ICsgeFIpICogJHtjb252SW5mby5pbldpZHRofSArIHhDKSAqICR7XG4gICAgICAgICAgICAgICAgICAgICAgIGNvbnZJbmZvLmluQ2hhbm5lbHN9ICsgY2hgIDpcbiAgICAgICAgICAgICAgICAgICBgKCh4RCAqICR7Y29udkluZm8uaW5IZWlnaHR9ICsgeFIpICogJHtcbiAgICAgICAgICAgICAgICAgICAgICAgY29udkluZm8uaW5XaWR0aH0gKyB4QykgKiAke2NvbnZJbmZvLmluQ2hhbm5lbHN9ICsgY2hgKSA6XG4gICAgICAgICAgICAgIGB3RCAqICR7ZWZmZWN0aXZlRmlsdGVySGVpZ2h0fSAqICR7ZWZmZWN0aXZlRmlsdGVyV2lkdGh9ICtcbiAgICAgICAgICAgICAgICAgICAgICB3UiAqICR7ZWZmZWN0aXZlRmlsdGVyV2lkdGh9ICsgd0NgfTtcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgc2V0T3V0cHV0KGZsb2F0KG1pbk1heFBvc2l0aW9uKSk7XG4gICAgICAgIH1cbiAgICAgIGA7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgY29tcGFyZU9wID0gJ21heCc7XG5cbiAgICBsZXQgcmV0dXJuVmFsdWUgPSBgJHtwb29sVHlwZX0oJHtwb29sVHlwZX0oJHtwb29sVHlwZX0oYCArXG4gICAgICAgICdtaW5NYXhWYWx1ZVswXSwgbWluTWF4VmFsdWVbMV0pLCBtaW5NYXhWYWx1ZVsyXSksIG1pbk1heFZhbHVlWzNdKSc7XG4gICAgaWYgKHBvb2xUeXBlID09PSAnYXZnJykge1xuICAgICAgLy8gVXNlIGBtYXgoY291bnQsIDEuMClgIGluc3RlYWQgb2YgYGNvdW50YCBpbiBjYXNlIGNvdW50ID09PSAwLjAuXG4gICAgICAvLyBJZiBjb3VudCA9PT0gMC4wLCBgYXZnVmFsdWVgIGlzIGFsd2F5cyAwLjAgYW5kIHdlIGNoYW5nZSBgY291bnRgJ3NcbiAgICAgIC8vIHZhbHVlIHRvIGF2b2lkIGRpdmlkaW5nIHplcm8uXG4gICAgICByZXR1cm5WYWx1ZSA9IGBhdmdWYWx1ZSAvIG1heChjb3VudCwgMS4wKWA7XG4gICAgfVxuXG4gICAgY29uc3QgZmlsdGVyV2lkdGhOZWFyZXN0VmVjNCA9IE1hdGguZmxvb3IoZmlsdGVyV2lkdGggLyA0KSAqIDQ7XG4gICAgY29uc3QgZmlsdGVyV2lkdGhWZWM0UmVtYWluZGVyID0gZmlsdGVyV2lkdGggJSA0O1xuXG4gICAgY29uc3QgdXBkYXRlU25pcHBldCA9IGBcbiAgICAgIGlmICgke2lzQXZnUG9vbH0pIHtcbiAgICAgICAgYXZnVmFsdWUgKz0gZG90KHZhbHVlcywgb25lcyk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBtaW5NYXhWYWx1ZSA9ICR7Y29tcGFyZU9wfSh2YWx1ZXMsIG1pbk1heFZhbHVlKTtcbiAgICAgIH1cbiAgICBgO1xuXG4gICAgdGhpcy51c2VyQ29kZSA9IGBcbiAgICAgIGNvbnN0IGl2ZWMzIHN0cmlkZXMgPVxuICAgICAgICBpdmVjMygke3N0cmlkZURlcHRofSwgJHtzdHJpZGVIZWlnaHR9LCAke3N0cmlkZVdpZHRofSk7XG4gICAgICBjb25zdCBpdmVjMyBwYWRzID0gaXZlYzMoJHtwYWRGcm9udH0sICR7cGFkVG9wfSwgJHtwYWRMZWZ0fSk7XG4gICAgICBjb25zdCBmbG9hdCBpbml0aWFsaXphdGlvblZhbHVlID0gJHtpbml0aWFsaXphdGlvblZhbHVlfTtcbiAgICAgIGNvbnN0IHZlYzQgb25lcyA9IHZlYzQoMS4wLCAxLjAsIDEuMCwgMS4wKTtcblxuICAgICAgZmxvYXQgY291bnQgPSAwLjA7XG5cbiAgICAgIGZsb2F0IGdldFZhbHVlKGludCBiYXRjaCwgaW50IHhELCBpbnQgeFIsIGludCB4QywgaW50IGNoKSB7XG4gICAgICAgIGlmICh4QyA8IDAgfHwgeEMgPj0gJHtjb252SW5mby5pbldpZHRofSkge1xuICAgICAgICAgIHJldHVybiBpbml0aWFsaXphdGlvblZhbHVlO1xuICAgICAgICB9XG4gICAgICAgIGNvdW50ICs9IDEuMDtcbiAgICAgICAgcmV0dXJuIGdldFgoYmF0Y2gsIHhELCB4UiwgeEMsIGNoKTtcbiAgICAgIH1cblxuICAgICAgdm9pZCBtYWluKCkge1xuICAgICAgICBpdmVjNSBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgaW50IGJhdGNoID0gY29vcmRzLng7XG4gICAgICAgIGludCBjaCA9IGNvb3Jkcy51O1xuXG4gICAgICAgIGl2ZWMzIHhDb3JuZXIgPSBpdmVjMyhjb29yZHMueSwgY29vcmRzLnosIGNvb3Jkcy53KSAqIHN0cmlkZXMgLSBwYWRzO1xuICAgICAgICBpbnQgeERDb3JuZXIgPSB4Q29ybmVyLng7XG4gICAgICAgIGludCB4UkNvcm5lciA9IHhDb3JuZXIueTtcbiAgICAgICAgaW50IHhDQ29ybmVyID0geENvcm5lci56O1xuXG4gICAgICAgIC8vIG1heC9taW4geCg/LCA/LCA/LCBkKSB0byBnZXQgeSh5RCwgeVIsIHlDLCBjaCkuXG4gICAgICAgIC8vID8gPSB0byBiZSBkZXRlcm1pbmVkXG4gICAgICAgIHZlYzQgbWluTWF4VmFsdWUgPSB2ZWM0KCR7aW5pdGlhbGl6YXRpb25WYWx1ZX0pO1xuICAgICAgICBmbG9hdCBhdmdWYWx1ZSA9IDAuMDtcbiAgICAgICAgY291bnQgPSAwLjA7XG5cbiAgICAgICAgZm9yIChpbnQgd0QgPSAwOyB3RCA8ICR7ZWZmZWN0aXZlRmlsdGVyRGVwdGh9O1xuICAgICAgICAgICAgd0QgKz0gJHtkaWxhdGlvbkRlcHRofSkge1xuICAgICAgICAgIGludCB4RCA9IHhEQ29ybmVyICsgd0Q7XG5cbiAgICAgICAgICBpZiAoeEQgPCAwIHx8IHhEID49ICR7Y29udkluZm8uaW5EZXB0aH0pIHtcbiAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGZvciAoaW50IHdSID0gMDsgd1IgPCAke2VmZmVjdGl2ZUZpbHRlckhlaWdodH07XG4gICAgICAgICAgICB3UiArPSAke2RpbGF0aW9uSGVpZ2h0fSkge1xuICAgICAgICAgICAgaW50IHhSID0geFJDb3JuZXIgKyB3UjtcblxuICAgICAgICAgICAgaWYgKHhSIDwgMCB8fCB4UiA+PSAke2NvbnZJbmZvLmluSGVpZ2h0fSkge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgZm9yIChpbnQgd0MgPSAwOyB3QyA8ICR7ZmlsdGVyV2lkdGhOZWFyZXN0VmVjNH07IHdDICs9IDQpIHtcbiAgICAgICAgICAgICAgaW50IHhDID0geENDb3JuZXIgKyB3QyAqICR7ZGlsYXRpb25XaWR0aH07XG5cbiAgICAgICAgICAgICAgdmVjNCB2YWx1ZXMgPSB2ZWM0KFxuICAgICAgICAgICAgICAgIGdldFZhbHVlKGJhdGNoLCB4RCwgeFIsIHhDLCBjaCksXG4gICAgICAgICAgICAgICAgZ2V0VmFsdWUoYmF0Y2gsIHhELCB4UiwgeEMgKyAke2RpbGF0aW9uV2lkdGh9LCBjaCksXG4gICAgICAgICAgICAgICAgZ2V0VmFsdWUoYmF0Y2gsIHhELCB4UiwgeEMgKyAyICogJHtkaWxhdGlvbldpZHRofSwgY2gpLFxuICAgICAgICAgICAgICAgIGdldFZhbHVlKGJhdGNoLCB4RCwgeFIsIHhDICsgMyAqICR7ZGlsYXRpb25XaWR0aH0sIGNoKVxuICAgICAgICAgICAgICApO1xuXG4gICAgICAgICAgICAgICR7dXBkYXRlU25pcHBldH1cbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgaW50IHhDID0geENDb3JuZXIgKyAke2ZpbHRlcldpZHRoTmVhcmVzdFZlYzR9O1xuICAgICAgICAgICAgaWYgKCR7ZmlsdGVyV2lkdGhWZWM0UmVtYWluZGVyID09PSAxfSkge1xuICAgICAgICAgICAgICB2ZWM0IHZhbHVlcyA9IHZlYzQoXG4gICAgICAgICAgICAgICAgZ2V0VmFsdWUoYmF0Y2gsIHhELCB4UiwgeEMsIGNoKSxcbiAgICAgICAgICAgICAgICBpbml0aWFsaXphdGlvblZhbHVlLFxuICAgICAgICAgICAgICAgIGluaXRpYWxpemF0aW9uVmFsdWUsXG4gICAgICAgICAgICAgICAgaW5pdGlhbGl6YXRpb25WYWx1ZVxuICAgICAgICAgICAgICApO1xuXG4gICAgICAgICAgICAgICR7dXBkYXRlU25pcHBldH1cbiAgICAgICAgICAgIH0gZWxzZSBpZiAoJHtmaWx0ZXJXaWR0aFZlYzRSZW1haW5kZXIgPT09IDJ9KSB7XG4gICAgICAgICAgICAgIHZlYzQgdmFsdWVzID0gdmVjNChcbiAgICAgICAgICAgICAgICBnZXRWYWx1ZShiYXRjaCwgeEQsIHhSLCB4QywgY2gpLFxuICAgICAgICAgICAgICAgIGdldFZhbHVlKGJhdGNoLCB4RCwgeFIsIHhDICsgJHtkaWxhdGlvbldpZHRofSwgY2gpLFxuICAgICAgICAgICAgICAgIGluaXRpYWxpemF0aW9uVmFsdWUsXG4gICAgICAgICAgICAgICAgaW5pdGlhbGl6YXRpb25WYWx1ZVxuICAgICAgICAgICAgICApO1xuXG4gICAgICAgICAgICAgICR7dXBkYXRlU25pcHBldH1cbiAgICAgICAgICAgIH0gZWxzZSBpZiAoJHtmaWx0ZXJXaWR0aFZlYzRSZW1haW5kZXIgPT09IDN9KSB7XG4gICAgICAgICAgICAgIHZlYzQgdmFsdWVzID0gdmVjNChcbiAgICAgICAgICAgICAgICBnZXRWYWx1ZShiYXRjaCwgeEQsIHhSLCB4QywgY2gpLFxuICAgICAgICAgICAgICAgIGdldFZhbHVlKGJhdGNoLCB4RCwgeFIsIHhDICsgJHtkaWxhdGlvbldpZHRofSwgY2gpLFxuICAgICAgICAgICAgICAgIGdldFZhbHVlKGJhdGNoLCB4RCwgeFIsIHhDICsgMiAqICR7ZGlsYXRpb25XaWR0aH0sIGNoKSxcbiAgICAgICAgICAgICAgICBpbml0aWFsaXphdGlvblZhbHVlXG4gICAgICAgICAgICAgICk7XG5cbiAgICAgICAgICAgICAgJHt1cGRhdGVTbmlwcGV0fVxuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBzZXRPdXRwdXQoJHtyZXR1cm5WYWx1ZX0pO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbn1cbiJdfQ==