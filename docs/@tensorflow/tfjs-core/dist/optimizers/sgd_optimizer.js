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
 * =============================================================================
 */
import { ENGINE } from '../engine';
import { keep, tidy } from '../globals';
import { add } from '../ops/add';
import { mul } from '../ops/mul';
import { scalar } from '../ops/scalar';
import { Optimizer } from './optimizer';
/** @doclink Optimizer */
export class SGDOptimizer extends Optimizer {
    constructor(learningRate) {
        super();
        this.learningRate = learningRate;
        this.setLearningRate(learningRate);
    }
    /** @nocollapse */
    static get className() {
        // Name matters for Python compatibility.
        // This is a getter instead of a property because when it's a property, it
        // prevents the entire class from being tree-shaken.
        return 'SGD';
    }
    applyGradients(variableGradients) {
        const varNames = Array.isArray(variableGradients) ?
            variableGradients.map(v => v.name) :
            Object.keys(variableGradients);
        varNames.forEach((name, i) => {
            const gradient = Array.isArray(variableGradients) ?
                variableGradients[i].tensor :
                variableGradients[name];
            if (gradient == null) {
                return;
            }
            const value = ENGINE.registeredVariables[name];
            tidy(() => {
                const newValue = add(mul(this.c, gradient), value);
                value.assign(newValue);
            });
        });
        this.incrementIterations();
    }
    /**
     * Sets the learning rate of the optimizer.
     */
    setLearningRate(learningRate) {
        this.learningRate = learningRate;
        if (this.c != null) {
            this.c.dispose();
        }
        this.c = keep(scalar(-learningRate));
    }
    dispose() {
        this.c.dispose();
    }
    async getWeights() {
        return [await this.saveIterations()];
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        if (weightValues.length !== 0) {
            throw new Error('SGD optimizer does not have settable weights.');
        }
    }
    getConfig() {
        return { 'learningRate': this.learningRate };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate']);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2dkX29wdGltaXplci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3B0aW1pemVycy9zZ2Rfb3B0aW1pemVyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLElBQUksRUFBRSxJQUFJLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDdEMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUMvQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFLckMsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUV0Qyx5QkFBeUI7QUFDekIsTUFBTSxPQUFPLFlBQWEsU0FBUSxTQUFTO0lBVXpDLFlBQXNCLFlBQW9CO1FBQ3hDLEtBQUssRUFBRSxDQUFDO1FBRFksaUJBQVksR0FBWixZQUFZLENBQVE7UUFFeEMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxZQUFZLENBQUMsQ0FBQztJQUNyQyxDQUFDO0lBWkQsa0JBQWtCO0lBQ2xCLE1BQU0sS0FBSyxTQUFTO1FBQ2xCLHlDQUF5QztRQUN6QywwRUFBMEU7UUFDMUUsb0RBQW9EO1FBQ3BELE9BQU8sS0FBSyxDQUFDO0lBQ2YsQ0FBQztJQVFELGNBQWMsQ0FBQyxpQkFBK0M7UUFDNUQsTUFBTSxRQUFRLEdBQUcsS0FBSyxDQUFDLE9BQU8sQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLENBQUM7WUFDL0MsaUJBQWlCLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDcEMsTUFBTSxDQUFDLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDO1FBQ25DLFFBQVEsQ0FBQyxPQUFPLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxFQUFFLEVBQUU7WUFDM0IsTUFBTSxRQUFRLEdBQUcsS0FBSyxDQUFDLE9BQU8sQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLENBQUM7Z0JBQy9DLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDO2dCQUM3QixpQkFBaUIsQ0FBQyxJQUFJLENBQUMsQ0FBQztZQUM1QixJQUFJLFFBQVEsSUFBSSxJQUFJLEVBQUU7Z0JBQ3BCLE9BQU87YUFDUjtZQUNELE1BQU0sS0FBSyxHQUFHLE1BQU0sQ0FBQyxtQkFBbUIsQ0FBQyxJQUFJLENBQUMsQ0FBQztZQUMvQyxJQUFJLENBQUMsR0FBRyxFQUFFO2dCQUNSLE1BQU0sUUFBUSxHQUFHLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsRUFBRSxRQUFRLENBQUMsRUFBRSxLQUFLLENBQUMsQ0FBQztnQkFDbkQsS0FBSyxDQUFDLE1BQU0sQ0FBQyxRQUFRLENBQUMsQ0FBQztZQUN6QixDQUFDLENBQUMsQ0FBQztRQUNMLENBQUMsQ0FBQyxDQUFDO1FBQ0gsSUFBSSxDQUFDLG1CQUFtQixFQUFFLENBQUM7SUFDN0IsQ0FBQztJQUVEOztPQUVHO0lBQ0gsZUFBZSxDQUFDLFlBQW9CO1FBQ2xDLElBQUksQ0FBQyxZQUFZLEdBQUcsWUFBWSxDQUFDO1FBQ2pDLElBQUksSUFBSSxDQUFDLENBQUMsSUFBSSxJQUFJLEVBQUU7WUFDbEIsSUFBSSxDQUFDLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztTQUNsQjtRQUNELElBQUksQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUM7SUFDdkMsQ0FBQztJQUVRLE9BQU87UUFDZCxJQUFJLENBQUMsQ0FBQyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ25CLENBQUM7SUFFUSxLQUFLLENBQUMsVUFBVTtRQUN2QixPQUFPLENBQUMsTUFBTSxJQUFJLENBQUMsY0FBYyxFQUFFLENBQUMsQ0FBQztJQUN2QyxDQUFDO0lBRVEsS0FBSyxDQUFDLFVBQVUsQ0FBQyxZQUEyQjtRQUNuRCxZQUFZLEdBQUcsTUFBTSxJQUFJLENBQUMsaUJBQWlCLENBQUMsWUFBWSxDQUFDLENBQUM7UUFDMUQsSUFBSSxZQUFZLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtZQUM3QixNQUFNLElBQUksS0FBSyxDQUFDLCtDQUErQyxDQUFDLENBQUM7U0FDbEU7SUFDSCxDQUFDO0lBRUQsU0FBUztRQUNQLE9BQU8sRUFBQyxjQUFjLEVBQUUsSUFBSSxDQUFDLFlBQVksRUFBQyxDQUFDO0lBQzdDLENBQUM7SUFFRCxrQkFBa0I7SUFDbEIsTUFBTSxDQUFVLFVBQVUsQ0FDdEIsR0FBK0IsRUFBRSxNQUFrQjtRQUNyRCxPQUFPLElBQUksR0FBRyxDQUFDLE1BQU0sQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDO0lBQ3pDLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge2tlZXAsIHRpZHl9IGZyb20gJy4uL2dsb2JhbHMnO1xuaW1wb3J0IHthZGR9IGZyb20gJy4uL29wcy9hZGQnO1xuaW1wb3J0IHttdWx9IGZyb20gJy4uL29wcy9tdWwnO1xuaW1wb3J0IHtzY2FsYXJ9IGZyb20gJy4uL29wcy9zY2FsYXInO1xuaW1wb3J0IHtDb25maWdEaWN0LCBTZXJpYWxpemFibGUsIFNlcmlhbGl6YWJsZUNvbnN0cnVjdG9yfSBmcm9tICcuLi9zZXJpYWxpemF0aW9uJztcbmltcG9ydCB7U2NhbGFyfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvciwgTmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5cbmltcG9ydCB7T3B0aW1pemVyfSBmcm9tICcuL29wdGltaXplcic7XG5cbi8qKiBAZG9jbGluayBPcHRpbWl6ZXIgKi9cbmV4cG9ydCBjbGFzcyBTR0RPcHRpbWl6ZXIgZXh0ZW5kcyBPcHRpbWl6ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGdldCBjbGFzc05hbWUoKSB7XG4gICAgLy8gTmFtZSBtYXR0ZXJzIGZvciBQeXRob24gY29tcGF0aWJpbGl0eS5cbiAgICAvLyBUaGlzIGlzIGEgZ2V0dGVyIGluc3RlYWQgb2YgYSBwcm9wZXJ0eSBiZWNhdXNlIHdoZW4gaXQncyBhIHByb3BlcnR5LCBpdFxuICAgIC8vIHByZXZlbnRzIHRoZSBlbnRpcmUgY2xhc3MgZnJvbSBiZWluZyB0cmVlLXNoYWtlbi5cbiAgICByZXR1cm4gJ1NHRCc7XG4gIH1cbiAgcHJvdGVjdGVkIGM6IFNjYWxhcjtcblxuICBjb25zdHJ1Y3Rvcihwcm90ZWN0ZWQgbGVhcm5pbmdSYXRlOiBudW1iZXIpIHtcbiAgICBzdXBlcigpO1xuICAgIHRoaXMuc2V0TGVhcm5pbmdSYXRlKGxlYXJuaW5nUmF0ZSk7XG4gIH1cblxuICBhcHBseUdyYWRpZW50cyh2YXJpYWJsZUdyYWRpZW50czogTmFtZWRUZW5zb3JNYXB8TmFtZWRUZW5zb3JbXSkge1xuICAgIGNvbnN0IHZhck5hbWVzID0gQXJyYXkuaXNBcnJheSh2YXJpYWJsZUdyYWRpZW50cykgP1xuICAgICAgICB2YXJpYWJsZUdyYWRpZW50cy5tYXAodiA9PiB2Lm5hbWUpIDpcbiAgICAgICAgT2JqZWN0LmtleXModmFyaWFibGVHcmFkaWVudHMpO1xuICAgIHZhck5hbWVzLmZvckVhY2goKG5hbWUsIGkpID0+IHtcbiAgICAgIGNvbnN0IGdyYWRpZW50ID0gQXJyYXkuaXNBcnJheSh2YXJpYWJsZUdyYWRpZW50cykgP1xuICAgICAgICAgIHZhcmlhYmxlR3JhZGllbnRzW2ldLnRlbnNvciA6XG4gICAgICAgICAgdmFyaWFibGVHcmFkaWVudHNbbmFtZV07XG4gICAgICBpZiAoZ3JhZGllbnQgPT0gbnVsbCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgICBjb25zdCB2YWx1ZSA9IEVOR0lORS5yZWdpc3RlcmVkVmFyaWFibGVzW25hbWVdO1xuICAgICAgdGlkeSgoKSA9PiB7XG4gICAgICAgIGNvbnN0IG5ld1ZhbHVlID0gYWRkKG11bCh0aGlzLmMsIGdyYWRpZW50KSwgdmFsdWUpO1xuICAgICAgICB2YWx1ZS5hc3NpZ24obmV3VmFsdWUpO1xuICAgICAgfSk7XG4gICAgfSk7XG4gICAgdGhpcy5pbmNyZW1lbnRJdGVyYXRpb25zKCk7XG4gIH1cblxuICAvKipcbiAgICogU2V0cyB0aGUgbGVhcm5pbmcgcmF0ZSBvZiB0aGUgb3B0aW1pemVyLlxuICAgKi9cbiAgc2V0TGVhcm5pbmdSYXRlKGxlYXJuaW5nUmF0ZTogbnVtYmVyKSB7XG4gICAgdGhpcy5sZWFybmluZ1JhdGUgPSBsZWFybmluZ1JhdGU7XG4gICAgaWYgKHRoaXMuYyAhPSBudWxsKSB7XG4gICAgICB0aGlzLmMuZGlzcG9zZSgpO1xuICAgIH1cbiAgICB0aGlzLmMgPSBrZWVwKHNjYWxhcigtbGVhcm5pbmdSYXRlKSk7XG4gIH1cblxuICBvdmVycmlkZSBkaXNwb3NlKCkge1xuICAgIHRoaXMuYy5kaXNwb3NlKCk7XG4gIH1cblxuICBvdmVycmlkZSBhc3luYyBnZXRXZWlnaHRzKCk6IFByb21pc2U8TmFtZWRUZW5zb3JbXT4ge1xuICAgIHJldHVybiBbYXdhaXQgdGhpcy5zYXZlSXRlcmF0aW9ucygpXTtcbiAgfVxuXG4gIG92ZXJyaWRlIGFzeW5jIHNldFdlaWdodHMod2VpZ2h0VmFsdWVzOiBOYW1lZFRlbnNvcltdKTogUHJvbWlzZTx2b2lkPiB7XG4gICAgd2VpZ2h0VmFsdWVzID0gYXdhaXQgdGhpcy5leHRyYWN0SXRlcmF0aW9ucyh3ZWlnaHRWYWx1ZXMpO1xuICAgIGlmICh3ZWlnaHRWYWx1ZXMubGVuZ3RoICE9PSAwKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ1NHRCBvcHRpbWl6ZXIgZG9lcyBub3QgaGF2ZSBzZXR0YWJsZSB3ZWlnaHRzLicpO1xuICAgIH1cbiAgfVxuXG4gIGdldENvbmZpZygpOiBDb25maWdEaWN0IHtcbiAgICByZXR1cm4geydsZWFybmluZ1JhdGUnOiB0aGlzLmxlYXJuaW5nUmF0ZX07XG4gIH1cblxuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIG92ZXJyaWRlIGZyb21Db25maWc8VCBleHRlbmRzIFNlcmlhbGl6YWJsZT4oXG4gICAgICBjbHM6IFNlcmlhbGl6YWJsZUNvbnN0cnVjdG9yPFQ+LCBjb25maWc6IENvbmZpZ0RpY3QpOiBUIHtcbiAgICByZXR1cm4gbmV3IGNscyhjb25maWdbJ2xlYXJuaW5nUmF0ZSddKTtcbiAgfVxufVxuIl19