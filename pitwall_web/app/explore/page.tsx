import TopBar from "../../components/TopBar";
import ExploreClient from "../../components/ExploreClient";

export default function ExplorePage() {
    return (
        <div className="shell">
            <TopBar />
            <div className="section">
                <div className="section-title">Explore Scenario</div>
                <ExploreClient />
            </div>
        </div>
    );
}
