import SwiftUI

/// Chip row for choosing which label the next placed spot gets (M4). Appears
/// above the toolbar while Spot mode is active so spots are colored correctly
/// as they're placed, not after.
struct LabelPicker: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        HStack(spacing: 6) {
            ForEach(SpotLabel.allCases, id: \.self) { label in
                Chip(label: label, isSelected: store.nextLabel == label) {
                    store.nextLabel = label
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Capsule().fill(.regularMaterial))
        .overlay(Capsule().strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
        .shadow(color: .black.opacity(0.12), radius: 10, y: 4)
    }
}

private struct Chip: View {
    let label: SpotLabel
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Circle().fill(label.color).frame(width: 8, height: 8)
                Text(label.rawValue).font(.system(size: 11.5, weight: isSelected ? .semibold : .regular))
            }
            .foregroundStyle(isSelected ? Color.primary : Color.secondary)
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(
                Capsule().fill(isSelected ? label.color.opacity(0.18) : Color.clear)
            )
            .overlay(
                Capsule().strokeBorder(isSelected ? label.color.opacity(0.5) : Color.clear, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}
